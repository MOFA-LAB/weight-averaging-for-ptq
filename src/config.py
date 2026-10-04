from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """The unified pretraining configuration is invalid."""


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CURVATURE_KEYS = {
    "batches",
    "micro_batch_size",
    "trace_probes",
    "lambda_max_iterations",
    "lambda_max_tolerance",
    "seed",
    "precision",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"missing configuration: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration must be a mapping: {path}")
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the only supported experiment schema.

    One YAML describes the complete run: a common AdamW trunk, one Stable
    continuation, the configured decay continuations, a bounded model-only
    checkpoint window for manual post-hoc averaging, trajectory measurements, and
    final Stable/decay measurements. Averaged endpoints are deliberately built by a
    separate command after training.
    """

    config_path = Path(path).expanduser().resolve()
    config = _read_yaml(config_path)
    _reject_unknown(
        config,
        {
            "project",
            "initialization",
            "runtime",
            "model_config",
            "data",
            "training",
            "optimizer",
            "post_fork",
            "evaluation",
        },
        "root",
    )
    model_path = Path(str(config.get("model_config", ""))).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    model_payload = _read_yaml(model_path.resolve())
    _reject_unknown(model_payload, {"model"}, "model config")
    model = _mapping(model_payload.get("model"), "model")
    _validate_model(model)

    config["model"] = model
    config["_config_path"] = str(config_path)
    config["_model_config_path"] = str(model_path.resolve())
    config["_project_root"] = str(config_path.parent.parent.resolve())
    _validate_experiment(config)
    return config


def _validate_model(model: dict[str, Any]) -> None:
    architecture = str(model.get("architecture", ""))
    common_keys = {
        "architecture",
        "vocab_size",
        "hidden_size",
        "ffn_dim",
        "num_hidden_layers",
        "num_attention_heads",
        "max_position_embeddings",
        "activation_function",
        "attention_dropout",
        "init_std",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "gradient_checkpointing",
        "expected_parameter_count",
    }
    architecture_keys = {
        "opt": {
            "word_embed_proj_dim",
            "dropout",
            "activation_dropout",
            "layerdrop",
            "do_layer_norm_before",
            "enable_bias",
        },
    }
    if architecture not in architecture_keys:
        raise ConfigError("model.architecture must be opt")
    _reject_unknown(
        model,
        common_keys | architecture_keys[architecture],
        "model",
    )
    for key in (
        "vocab_size",
        "hidden_size",
        "ffn_dim",
        "num_hidden_layers",
        "num_attention_heads",
        "max_position_embeddings",
    ):
        model[key] = _positive_int(model.get(key), f"model.{key}")
    if "expected_parameter_count" in model:
        model["expected_parameter_count"] = _positive_int(
            model["expected_parameter_count"], "model.expected_parameter_count"
        )
    if model["hidden_size"] % model["num_attention_heads"] != 0:
        raise ConfigError(
            "model.hidden_size must be divisible by model.num_attention_heads"
        )
    if not str(model.get("activation_function", "")).strip():
        raise ConfigError("model.activation_function is required")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        model[key] = _nonnegative_int(model.get(key), f"model.{key}")
        if model[key] >= model["vocab_size"]:
            raise ConfigError(f"model.{key} must be smaller than model.vocab_size")
    dropout_keys = ["attention_dropout"]
    model["word_embed_proj_dim"] = _positive_int(
        model.get("word_embed_proj_dim"), "model.word_embed_proj_dim"
    )
    dropout_keys.extend(("dropout", "activation_dropout", "layerdrop"))
    for key in dropout_keys:
        value = _finite_float(model.get(key), f"model.{key}")
        if not 0.0 <= value < 1.0:
            raise ConfigError(f"model.{key} must lie in [0,1)")
    if _finite_float(model.get("init_std"), "model.init_std") <= 0.0:
        raise ConfigError("model.init_std must be positive")
    boolean_keys = ["gradient_checkpointing", "do_layer_norm_before", "enable_bias"]
    for key in boolean_keys:
        if not isinstance(model.get(key), bool):
            raise ConfigError(f"model.{key} must be boolean")


def _validate_experiment(config: dict[str, Any]) -> None:
    project = _mapping(config.get("project"), "project")
    _reject_unknown(project, {"name", "output_dir", "seed", "resume"}, "project")
    _safe_name(project.get("name"), "project.name")
    if not str(project.get("output_dir", "")).strip():
        raise ConfigError("project.output_dir is required")
    project["seed"] = _nonnegative_int(project.get("seed"), "project.seed")
    if not isinstance(project.get("resume"), bool):
        raise ConfigError("project.resume must be boolean")

    initialization = _mapping(config.get("initialization"), "initialization")
    _reject_unknown(initialization, {"mode", "source_run"}, "initialization")
    mode = initialization.get("mode")
    if mode not in {"scratch", "fork"}:
        raise ConfigError("initialization.mode must be scratch or fork")
    has_source = "source_run" in initialization
    if mode == "scratch":
        if has_source:
            raise ConfigError(
                "initialization.source_run is forbidden when mode=scratch"
            )
    else:
        if not has_source or not str(initialization.get("source_run", "")).strip():
            raise ConfigError("initialization.source_run is required when mode=fork")
        initialization["source_run"] = str(initialization["source_run"])

    runtime = _mapping(config.get("runtime"), "runtime")
    _reject_unknown(runtime, {"gpu_ids", "device"}, "runtime")
    runtime.setdefault("device", "cuda")
    if runtime["device"] not in {"cuda", "cpu"}:
        raise ConfigError("runtime.device must be cuda or cpu")
    gpu_ids = runtime.get("gpu_ids")
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ConfigError("runtime.gpu_ids must be a non-empty list")
    runtime["gpu_ids"] = [
        _nonnegative_int(value, f"runtime.gpu_ids[{index}]")
        for index, value in enumerate(gpu_ids)
    ]
    if len(runtime["gpu_ids"]) != len(set(runtime["gpu_ids"])):
        raise ConfigError("runtime.gpu_ids must be distinct")

    data = _mapping(config.get("data"), "data")
    _reject_unknown(
        data,
        {
            "root",
            "manifest",
            "train_file",
            "validation_file",
            "storage_dtype",
            "tokenizer_vocab_size",
            "preparation",
        },
        "data",
    )
    for key in ("root", "manifest", "train_file", "validation_file"):
        if not str(data.get(key, "")).strip():
            raise ConfigError(f"data.{key} is required")
    if data.get("storage_dtype") != "int32":
        raise ConfigError("data.storage_dtype must be int32")
    data["tokenizer_vocab_size"] = _positive_int(
        data.get("tokenizer_vocab_size"), "data.tokenizer_vocab_size"
    )
    if data["tokenizer_vocab_size"] > int(config["model"]["vocab_size"]):
        raise ConfigError("data.tokenizer_vocab_size must not exceed model.vocab_size")
    data["preparation"] = _normalize_data_preparation(data.get("preparation"))

    training = _mapping(config.get("training"), "training")
    _reject_unknown(
        training,
        {
            "sequence_length",
            "epochs",
            "train_token_budget",
            "micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "expected_world_size",
            "precision",
            "max_grad_norm",
            "warmup_ratio",
            "stable_ratio",
            "decay_ratio",
            "log_interval",
            "recovery_checkpoint_interval",
            "deterministic_algorithms",
            "tf32",
        },
        "training",
    )
    for key in (
        "sequence_length",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "expected_world_size",
        "log_interval",
        "recovery_checkpoint_interval",
    ):
        training[key] = _positive_int(training.get(key), f"training.{key}")
    if training["expected_world_size"] != len(runtime["gpu_ids"]):
        raise ConfigError(
            "training.expected_world_size must equal len(runtime.gpu_ids)"
        )
    if training["sequence_length"] > int(config["model"]["max_position_embeddings"]):
        raise ConfigError(
            "training.sequence_length must not exceed model.max_position_embeddings"
        )
    if training.get("precision") not in {"bf16_mixed", "fp32"}:
        raise ConfigError("training.precision must be bf16_mixed or fp32")
    if _finite_float(training.get("epochs"), "training.epochs") != 1.0:
        raise ConfigError("training.epochs must be exactly 1.0")
    if _finite_float(training.get("max_grad_norm"), "training.max_grad_norm") <= 0.0:
        raise ConfigError("training.max_grad_norm must be positive")
    ratios = [
        _finite_float(training.get(name), f"training.{name}")
        for name in ("warmup_ratio", "stable_ratio", "decay_ratio")
    ]
    if any(value <= 0.0 for value in ratios) or not math.isclose(
        math.fsum(ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ConfigError("training phase ratios must be positive and sum to one")
    if "train_token_budget" in training:
        budget = _positive_int(
            training.get("train_token_budget"), "training.train_token_budget"
        )
        if budget % int(training["sequence_length"]) != 0:
            raise ConfigError(
                "training.train_token_budget must be divisible by training.sequence_length"
            )
        minimum = (
            int(training["sequence_length"])
            * int(training["micro_batch_size_per_gpu"])
            * int(training["gradient_accumulation_steps"])
            * int(training["expected_world_size"])
        )
        if budget < minimum:
            raise ConfigError(
                "training.train_token_budget must cover one global optimizer update"
            )
        prepared_minimum = int(data["preparation"].get("minimum_train_tokens", budget))
        if budget > prepared_minimum:
            raise ConfigError(
                "training.train_token_budget must not exceed data.preparation.minimum_train_tokens"
            )
        training["train_token_budget"] = budget
    for key in ("deterministic_algorithms", "tf32"):
        if not isinstance(training.get(key), bool):
            raise ConfigError(f"training.{key} must be boolean")

    optimizer = _mapping(config.get("optimizer"), "optimizer")
    _reject_unknown(
        optimizer,
        {"name", "learning_rate", "betas", "eps", "weight_decay"},
        "optimizer",
    )
    if optimizer.get("name") != "adamw":
        raise ConfigError("optimizer.name must be adamw")
    if _finite_float(optimizer.get("learning_rate"), "optimizer.learning_rate") <= 0.0:
        raise ConfigError("optimizer.learning_rate must be positive")
    betas = optimizer.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ConfigError("optimizer.betas must contain two values")
    optimizer["betas"] = [
        _probability(value, f"optimizer.betas[{index}]")
        for index, value in enumerate(betas)
    ]
    if _finite_float(optimizer.get("eps"), "optimizer.eps") <= 0.0:
        raise ConfigError("optimizer.eps must be positive")
    if _finite_float(optimizer.get("weight_decay"), "optimizer.weight_decay") < 0.0:
        raise ConfigError("optimizer.weight_decay must be non-negative")

    post_fork = _mapping(config.get("post_fork"), "post_fork")
    _reject_unknown(
        post_fork,
        {"selected_branches", "checkpoint_snapshots", "stable", "decay"},
        "post_fork",
    )
    snapshots = _mapping(
        post_fork.get("checkpoint_snapshots"), "post_fork.checkpoint_snapshots"
    )
    _reject_unknown(
        snapshots,
        {"interval", "keep_last"},
        "post_fork.checkpoint_snapshots",
    )
    snapshots["interval"] = _positive_int(
        snapshots.get("interval"), "post_fork.checkpoint_snapshots.interval"
    )
    snapshots["keep_last"] = _positive_int(
        snapshots.get("keep_last"), "post_fork.checkpoint_snapshots.keep_last"
    )
    stable = _mapping(post_fork.get("stable"), "post_fork.stable")
    _reject_unknown(stable, {"schedule"}, "post_fork.stable")
    if stable.get("schedule") != "constant":
        raise ConfigError("post_fork.stable.schedule must be constant")

    decay = _mapping(post_fork.get("decay"), "post_fork.decay")
    _reject_unknown(decay, {"branches"}, "post_fork.decay")
    branches = decay.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ConfigError("post_fork.decay.branches must be a non-empty list")
    normalized_branches: list[dict[str, Any]] = []
    names: set[str] = {"stable", "trunk", "fork"}
    specs: set[tuple[float, float, float]] = set()
    continuation_ratio = float(training["decay_ratio"])
    for index, raw in enumerate(branches):
        branch = _mapping(raw, f"post_fork.decay.branches[{index}]")
        _reject_unknown(
            branch,
            {
                "name",
                "schedule",
                "stable_ratio_after_fork",
                "decay_ratio",
                "final_lr_ratio",
            },
            f"post_fork.decay.branches[{index}]",
        )
        name = _safe_name(branch.get("name"), f"post_fork.decay.branches[{index}].name")
        if name in names:
            raise ConfigError(f"duplicate or reserved branch name: {name}")
        names.add(name)
        if branch.get("schedule") != "stable_then_cosine_decay":
            raise ConfigError(f"decay branch {name} must use stable_then_cosine_decay")
        stable_ratio = _nonnegative_float(
            branch.get("stable_ratio_after_fork"),
            f"post_fork.decay.branches[{index}].stable_ratio_after_fork",
        )
        decay_ratio = _finite_float(
            branch.get("decay_ratio"),
            f"post_fork.decay.branches[{index}].decay_ratio",
        )
        if decay_ratio <= 0.0:
            raise ConfigError(f"decay branch {name} must have positive decay_ratio")
        if not math.isclose(
            stable_ratio + decay_ratio,
            continuation_ratio,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ConfigError(
                f"decay branch {name} must cover training.decay_ratio={continuation_ratio}"
            )
        final_ratio = _unit_interval(
            branch.get("final_lr_ratio"),
            f"post_fork.decay.branches[{index}].final_lr_ratio",
        )
        signature = (stable_ratio, decay_ratio, final_ratio)
        if signature in specs:
            raise ConfigError(f"duplicate decay schedule: {name}")
        specs.add(signature)
        normalized_branches.append(
            {
                "name": name,
                "schedule": "stable_then_cosine_decay",
                "stable_ratio_after_fork": stable_ratio,
                "decay_ratio": decay_ratio,
                "final_lr_ratio": final_ratio,
            }
        )
    decay["branches"] = normalized_branches

    selection = post_fork.get("selected_branches")
    configured_order = ["stable", *[branch["name"] for branch in normalized_branches]]
    if selection == "all":
        post_fork["selected_branches"] = configured_order
    elif isinstance(selection, list) and selection:
        if any(not isinstance(item, str) or not item for item in selection):
            raise ConfigError(
                "post_fork.selected_branches items must be non-empty strings"
            )
        if len(selection) != len(set(selection)):
            raise ConfigError("post_fork.selected_branches must not contain duplicates")
        unknown = sorted(set(selection) - set(configured_order))
        if unknown:
            raise ConfigError(
                "post_fork.selected_branches contains unknown branches: "
                + ", ".join(unknown)
            )
        selected = set(selection)
        post_fork["selected_branches"] = [
            branch for branch in configured_order if branch in selected
        ]
    else:
        raise ConfigError("post_fork.selected_branches must be all or a non-empty list")

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    _reject_unknown(evaluation, {"validation", "trajectory", "final"}, "evaluation")
    validation = _mapping(evaluation.get("validation"), "evaluation.validation")
    _reject_unknown(
        validation,
        {"num_blocks", "micro_batch_size", "precision"},
        "evaluation.validation",
    )
    validation["num_blocks"] = _positive_int(
        validation.get("num_blocks"), "evaluation.validation.num_blocks"
    )
    validation["micro_batch_size"] = _positive_int(
        validation.get("micro_batch_size"),
        "evaluation.validation.micro_batch_size",
    )
    if validation["num_blocks"] % validation["micro_batch_size"] != 0:
        raise ConfigError(
            "evaluation.validation.num_blocks must be divisible by micro_batch_size"
        )
    if validation.get("precision") not in {"fp32", "bf16_mixed"}:
        raise ConfigError("evaluation.validation.precision must be fp32 or bf16_mixed")

    trajectory = _mapping(evaluation.get("trajectory"), "evaluation.trajectory")
    _reject_unknown(
        trajectory,
        {"interval", "include_fork", "include_final", "curvature"},
        "evaluation.trajectory",
    )
    trajectory["interval"] = _positive_int(
        trajectory.get("interval"), "evaluation.trajectory.interval"
    )
    for key in ("include_fork", "include_final"):
        if not isinstance(trajectory.get(key), bool):
            raise ConfigError(f"evaluation.trajectory.{key} must be boolean")
    trajectory["curvature"] = _normalize_curvature(
        trajectory.get("curvature"), "evaluation.trajectory.curvature"
    )

    final = _mapping(evaluation.get("final"), "evaluation.final")
    _reject_unknown(final, {"output_directory", "curvature"}, "evaluation.final")
    final["output_directory"] = _safe_relative_path(
        final.get("output_directory"), "evaluation.final.output_directory"
    )
    final["curvature"] = _normalize_curvature(
        final.get("curvature"), "evaluation.final.curvature"
    )


def _normalize_curvature(value: Any, label: str) -> dict[str, Any]:
    group = _mapping(value, label)
    _reject_unknown(group, {"enabled", *_CURVATURE_KEYS}, label)
    if "enabled" not in group:
        raise ConfigError(f"missing {label} key: enabled")
    if not isinstance(group["enabled"], bool):
        raise ConfigError(f"{label}.enabled must be boolean")
    if not group["enabled"]:
        if set(group) != {"enabled"}:
            extras = ", ".join(sorted(set(group) - {"enabled"}))
            raise ConfigError(
                f"{label} must contain only enabled=false; remove: {extras}"
            )
        return {"enabled": False}

    missing = sorted(_CURVATURE_KEYS - set(group))
    if missing:
        raise ConfigError(f"missing {label} keys: {', '.join(missing)}")
    normalized = {
        "enabled": True,
        "batches": _positive_int(group["batches"], f"{label}.batches"),
        "micro_batch_size": _positive_int(
            group["micro_batch_size"], f"{label}.micro_batch_size"
        ),
        "trace_probes": _positive_int(group["trace_probes"], f"{label}.trace_probes"),
        "lambda_max_iterations": _positive_int(
            group["lambda_max_iterations"], f"{label}.lambda_max_iterations"
        ),
        "lambda_max_tolerance": _nonnegative_float(
            group["lambda_max_tolerance"], f"{label}.lambda_max_tolerance"
        ),
        "seed": _nonnegative_int(group["seed"], f"{label}.seed"),
        "precision": str(group["precision"]),
    }
    if normalized["precision"] not in {"fp32", "bf16_mixed"}:
        raise ConfigError(f"{label}.precision must be fp32 or bf16_mixed")
    return normalized


def _normalize_data_preparation(value: Any) -> dict[str, Any]:
    label = "data.preparation"
    group = _mapping(value, label)
    if not isinstance(group.get("prepare_on_launch"), bool):
        raise ConfigError(f"{label}.prepare_on_launch must be boolean")
    prepare_on_launch = bool(group["prepare_on_launch"])
    mode = str(group.get("mode", ""))
    if mode == "verify_existing":
        _reject_unknown(group, {"mode", "prepare_on_launch"}, label)
        return {"mode": mode, "prepare_on_launch": prepare_on_launch}
    if mode != "fineweb_edu_local_snapshot":
        raise ConfigError(
            "data.preparation.mode must be verify_existing or fineweb_edu_local_snapshot"
        )
    allowed = {
        "mode",
        "raw_dir",
        "minimum_train_tokens",
        "validation_tokens",
        "dataset_revision",
        "tokenizer_assets_dir",
        "prepare_on_launch",
    }
    _reject_unknown(group, allowed, label)
    for key in ("raw_dir", "dataset_revision", "tokenizer_assets_dir"):
        if not str(group.get(key, "")).strip():
            raise ConfigError(f"{label}.{key} is required")
    minimum = _positive_int(
        group.get("minimum_train_tokens"), f"{label}.minimum_train_tokens"
    )
    validation = _positive_int(
        group.get("validation_tokens"), f"{label}.validation_tokens"
    )
    revision = str(group["dataset_revision"])
    return {
        "mode": mode,
        "prepare_on_launch": prepare_on_launch,
        "raw_dir": str(group["raw_dir"]),
        "minimum_train_tokens": minimum,
        "validation_tokens": validation,
        "dataset_revision": revision,
        "tokenizer_assets_dir": str(group["tokenizer_assets_dir"]),
    }


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _reject_unknown(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(f"unsupported {label} keys: {', '.join(unknown)}")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{label} must be a non-negative integer")
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{label} must be finite")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ConfigError(f"{label} must be non-negative")
    return result


def _unit_interval(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{label} must lie in [0,1]")
    return result


def _probability(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 <= result < 1.0:
        raise ConfigError(f"{label} must lie in [0,1)")
    return result


def _safe_name(value: Any, label: str) -> str:
    result = str(value or "")
    if not _SAFE_NAME.fullmatch(result):
        raise ConfigError(f"{label} must be a safe non-empty name")
    return result


def _safe_relative_path(value: Any, label: str) -> str:
    raw = str(value or "")
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ConfigError(f"{label} must be a safe relative path")
    if path.parts[0] in {
        "branches",
        "checkpoints",
        "resolved_config.json",
        "final_endpoints_manifest.json",
        "COMPLETED",
    }:
        raise ConfigError(f"{label} overlaps a reserved run artifact")
    return path.as_posix()
