from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# A file-path launch (``python tools/build_weighted_endpoints.py``) places only
# ``tools/`` on sys.path.  The project root is required for the non-packaged
# training entrypoint and must take precedence over any stale editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.data import load_corpus
from src.distributed import DistributedContext
from src.model import build_model
from src.utils import parameter_norm, seed_everything
from train import _evaluate_curvature_if_enabled, evaluate_validation

SNAPSHOT_STEP_RE = re.compile(r"step_(\d+)\.pt$")
SUPPORTED_AVERAGING_METHODS = frozenset({"lawa", "wma", "lnwa"})


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def durable_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config must be a YAML mapping")
    return payload


def _tensor_alias_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        str(tensor.dtype),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        tuple(int(x) for x in tensor.shape),
        tuple(int(x) for x in tensor.stride()),
    )


def alias_map(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    canonical_by_sig: dict[tuple[Any, ...], str] = {}
    result: dict[str, str] = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"state[{name!r}] is not a tensor")
        sig = _tensor_alias_signature(value)
        result[str(name)] = canonical_by_sig.setdefault(sig, str(name))
    return result


def assert_compatible(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor], label: str
) -> None:
    if list(reference) != list(candidate):
        raise RuntimeError(f"{label}: state_dict keys differ")
    for name, ref in reference.items():
        cur = candidate[name]
        if ref.shape != cur.shape or ref.dtype != cur.dtype:
            raise RuntimeError(
                f"{label}: tensor metadata differs for {name}: "
                f"{tuple(ref.shape)}/{ref.dtype} vs {tuple(cur.shape)}/{cur.dtype}"
            )


def extract_state(
    payload: Any, path: Path, expected_step: int | None = None
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Accept the model-only stable snapshot envelope, endpoint envelope, or raw state_dict."""
    metadata: dict[str, Any] = {}
    if (
        isinstance(payload, Mapping)
        and "model" in payload
        and isinstance(payload["model"], Mapping)
    ):
        state = payload["model"]
        if "metadata" in payload and isinstance(payload["metadata"], Mapping):
            metadata.update(dict(payload["metadata"]))
        for key in ("step", "branch"):
            if key in payload:
                metadata.setdefault(key, payload[key])
    elif (
        isinstance(payload, Mapping)
        and payload
        and all(torch.is_tensor(v) for v in payload.values())
    ):
        state = payload
    else:
        raise RuntimeError(f"unsupported checkpoint payload: {path}")

    ordered = OrderedDict((str(k), v) for k, v in state.items())
    if not ordered or not all(torch.is_tensor(v) for v in ordered.values()):
        raise RuntimeError(f"invalid model state in {path}")
    if (
        expected_step is not None
        and "step" in metadata
        and int(metadata["step"]) != int(expected_step)
    ):
        raise RuntimeError(
            f"checkpoint step mismatch: filename={expected_step}, payload={metadata['step']}, path={path}"
        )
    return ordered, metadata


def load_snapshot(
    path: Path, expected_step: int | None = None
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return extract_state(payload, path, expected_step)


def discover_snapshots(cfg: Mapping[str, Any]) -> tuple[list[Path], list[int], int, int]:
    stable_dir = Path(str(cfg["stable_dir"])).expanduser().resolve()
    pattern = str(cfg.get("checkpoint_glob", "step_*.pt"))
    found: list[tuple[int, Path]] = []
    for path in stable_dir.glob(pattern):
        match = SNAPSHOT_STEP_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda x: x[0])
    if not found:
        raise FileNotFoundError(f"no snapshots matched {stable_dir / pattern}")

    m = int(cfg.get("M", 10))
    retained_k = len(found)
    if m < 2 or m > retained_k:
        raise ValueError(f"input.M must satisfy 2 <= M <= {retained_k} saved snapshots; got {m}")
    found = found[-m:]
    steps = [s for s, _ in found]
    paths = [p for _, p in found]

    diffs = [steps[index + 1] - steps[index] for index in range(len(steps) - 1)]
    if any(x <= 0 for x in diffs):
        raise RuntimeError(f"snapshot steps are not strictly increasing: {steps}")
    configured = cfg.get("checkpoint_interval")
    inferred = diffs[0]
    require_uniform = bool(cfg.get("require_uniform_spacing", True))
    if require_uniform and any(x != inferred for x in diffs):
        raise RuntimeError(f"snapshot spacing is not uniform: steps={steps}, diffs={diffs}")
    if configured is not None and int(configured) != inferred:
        raise RuntimeError(
            f"checkpoint interval mismatch: configured={configured}, inferred={inferred}, steps={steps}"
        )

    return paths, steps, inferred, retained_k


def distances_from_latest(steps: Sequence[int], interval: int) -> list[float]:
    latest = int(steps[-1])
    return [(latest - int(step)) / float(interval) for step in steps]


def normalize_weights(weights: Sequence[float]) -> list[float]:
    total = math.fsum(float(x) for x in weights)
    if not math.isfinite(total) or abs(total) < 1e-15:
        raise ValueError(f"cannot normalize weights with sum={total}")
    return [float(x) / total for x in weights]


def lawa_weights(m: int) -> list[float]:
    return [1.0 / m] * m


def wma_weights(m: int) -> list[float]:
    # Linear LAWA / WMA: oldest gets 1, newest gets M.
    raw = [float(j + 1) for j in range(m)]
    return normalize_weights(raw)


def lnwa_weights(distances: Sequence[float], lag_penalty: float) -> list[float]:
    """Solve min ||w||_2^2 + lambda (d^T w)^2 on the probability simplex."""

    penalty = float(lag_penalty)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("LNWA lag penalty must be finite and >= 0")
    d = [float(value) for value in distances]
    if len(d) < 2 or not all(math.isfinite(value) and value >= 0.0 for value in d):
        raise ValueError("LNWA requires at least two finite non-negative distances")
    if len(set(d)) != len(d):
        raise ValueError("LNWA checkpoint distances must be distinct")

    # KKT active-set solution.  For a fixed active support I,
    # Q_I = I + lambda d_I d_I^T and
    # w_I = Q_I^{-1} 1 / (1^T Q_I^{-1} 1).  Non-positive coordinates are
    # necessarily the oldest ones and are removed by complementary slackness.
    active = list(range(len(d)))
    tolerance = 1e-14
    while True:
        active_distances = [d[index] for index in active]
        count = len(active)
        s1 = math.fsum(active_distances)
        s2 = math.fsum(value * value for value in active_distances)
        denominator = count + penalty * (count * s2 - s1 * s1)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("LNWA active-set system is not positive definite")
        candidate = [
            (1.0 + penalty * s2 - penalty * s1 * value) / denominator for value in active_distances
        ]
        keep = [index for index, weight in zip(active, candidate) if weight > tolerance]
        if keep == active:
            weights = [0.0] * len(d)
            for index, weight in zip(active, candidate):
                weights[index] = float(weight)
            total = math.fsum(weights)
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"LNWA weights do not sum to one: {total}")
            return weights
        if not keep:
            keep = [min(active, key=d.__getitem__)]
        active = keep


def weight_stats(weights: Sequence[float], distances: Sequence[float]) -> dict[str, float]:
    w = [float(x) for x in weights]
    d = [float(x) for x in distances]
    total = math.fsum(w)
    m1 = math.fsum(wi * di for wi, di in zip(w, d))
    r = math.fsum(wi * wi for wi in w)
    neg = math.fsum(-wi for wi in w if wi < 0.0)
    return {
        "sum": total,
        "first_temporal_moment": m1,
        "noise_factor_R": r,
        "effective_sample_size": (1.0 / r if r > 0.0 else float("inf")),
        "negative_weight_mass": neg,
        "min_weight": min(w),
        "max_weight": max(w),
    }


def _float_slug(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def linear_merge(
    paths: Sequence[Path],
    steps: Sequence[int],
    weights: Sequence[float],
) -> OrderedDict[str, torch.Tensor]:
    if not (len(paths) == len(steps) == len(weights)) or not paths:
        raise ValueError("paths/steps/weights must have the same nonzero length")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=2e-9):
        raise ValueError(f"endpoint weights must sum to 1; got {math.fsum(weights)}")

    result: OrderedDict[str, torch.Tensor] | None = None
    reference: OrderedDict[str, torch.Tensor] | None = None
    aliases: dict[str, str] | None = None

    for idx, (path, step, weight) in enumerate(zip(paths, steps, weights)):
        state, metadata = load_snapshot(path, step)
        if metadata.get("branch", "stable") not in {"stable", "trunk"}:
            raise ValueError(f"expected a Stable snapshot, got branch={metadata['branch']!r}: {path}")
        if reference is None:
            reference = state
            aliases = alias_map(state)
            result = OrderedDict()
            for name, value in state.items():
                canonical = aliases[name]
                if canonical != name:
                    continue
                if value.is_floating_point():
                    result[name] = value.to(dtype=torch.float32, copy=True).mul_(float(weight))
                else:
                    # buffers are replaced from the latest checkpoint below
                    result[name] = value.clone()
        else:
            assert result is not None and aliases is not None
            assert_compatible(reference, state, f"checkpoint step {step}")
            for name, value in state.items():
                if aliases[name] != name or not value.is_floating_point():
                    continue
                result[name].add_(value.to(dtype=torch.float32), alpha=float(weight))

        if idx + 1 == len(paths):
            assert result is not None and aliases is not None
            for name, value in state.items():
                if aliases[name] == name and not value.is_floating_point():
                    result[name] = value.clone()

        del state
        gc.collect()

    assert result is not None and reference is not None and aliases is not None
    for name, canonical in aliases.items():
        if name != canonical:
            result[name] = result[canonical]
    return OrderedDict((name, result[name]) for name in reference)


def save_endpoint(
    path: Path, state: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    durable_torch_save(
        path,
        {
            "metadata": dict(metadata),
            "model": OrderedDict(state),
        },
    )
    return {"bytes": int(path.stat().st_size)}


def _source_resolved_path(output_cfg: Mapping[str, Any], source_root: Path) -> Path:
    explicit = output_cfg.get("source_resolved_config")
    if explicit is None:
        return source_root / "resolved_config.json"
    if not isinstance(explicit, str) or not explicit.strip():
        raise ValueError("output.source_resolved_config must be a non-empty path")
    return Path(explicit).expanduser().resolve()


def load_source_resolved(output_cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    raw = output_cfg.get("source_run_dir")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("output.source_run_dir is required")
    source_root = Path(raw).expanduser().resolve()
    path = _source_resolved_path(output_cfg, source_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing source resolved config: {path}; set output.source_run_dir to the training run"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"source resolved config is not a mapping: {path}")
    for key in ("model", "data", "training"):
        if not isinstance(payload.get(key), dict):
            raise TypeError(f"source resolved config lacks a {key} mapping: {path}")
    return source_root, payload


def prepare_evaluator(
    config: Mapping[str, Any],
    source_resolved: Mapping[str, Any],
) -> tuple[torch.nn.Module, Any, torch.device, DistributedContext]:
    runtime = config.get("runtime")
    evaluation = config.get("evaluation")
    if not isinstance(runtime, Mapping) or not isinstance(evaluation, Mapping):
        raise TypeError("offline averaging config requires runtime and evaluation mappings")
    device_type = str(runtime.get("device", "cuda"))
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; set runtime.device: cpu for CPU evaluation")
        gpu_id = int(runtime.get("gpu_id", 0))
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
    elif device_type == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("runtime.device must be cuda or cpu")
    deterministic = bool(runtime.get("deterministic_algorithms", True))
    seed_everything(int(runtime.get("seed", 20260722)), deterministic)
    tf32 = bool(runtime.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    model_config = source_resolved.get("model")
    data_config = source_resolved.get("data")
    training = source_resolved.get("training")
    if not all(isinstance(item, Mapping) for item in (model_config, data_config, training)):
        raise ValueError("source resolved config lacks model/data/training")
    corpus = load_corpus(dict(data_config), int(training["sequence_length"]))
    model = build_model(dict(model_config), int(runtime.get("seed", 20260722)))
    model.to(device=device, dtype=torch.float32)
    return model, corpus, device, DistributedContext(rank=0, local_rank=0, world_size=1)


def evaluate_endpoint_state(
    *,
    name: str,
    state: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    corpus: Any,
    evaluation: Mapping[str, Any],
    device: torch.device,
    context: DistributedContext,
) -> dict[str, Any]:
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"averaged endpoint state mismatch for {name}")
    validation = evaluation.get("validation")
    curvature = evaluation.get("curvature")
    if not isinstance(validation, Mapping) or not isinstance(curvature, Mapping):
        raise TypeError("evaluation requires validation and curvature mappings")
    curvature_enabled = curvature.get("enabled")
    if not isinstance(curvature_enabled, bool):
        raise TypeError("evaluation.curvature.enabled must be boolean")
    result: dict[str, Any] = {
        "name": name,
        "curvature_enabled": curvature_enabled,
    }
    result.update(evaluate_validation(model, corpus, validation, device, context))
    result.update(_evaluate_curvature_if_enabled(model, corpus, curvature, device, context))
    result["parameter_norm"] = parameter_norm(model)
    return result


def build_method_weights(
    config: Mapping[str, Any],
    steps: Sequence[int],
    interval: int,
) -> list[dict[str, Any]]:
    methods = config.get("methods")
    if not isinstance(methods, Mapping):
        raise TypeError("config.methods must be a mapping")
    if not all(isinstance(name, str) for name in methods):
        raise TypeError("config.methods keys must be strings")

    unsupported = sorted(set(methods) - SUPPORTED_AVERAGING_METHODS)
    if unsupported:
        raise ValueError(
            f"unsupported averaging methods: {unsupported}; "
            f"supported={sorted(SUPPORTED_AVERAGING_METHODS)}"
        )

    allowed_settings = {
        "lawa": {"enabled"},
        "wma": {"enabled"},
        "lnwa": {"enabled", "lag_penalties"},
    }

    def method_settings(name: str) -> tuple[Mapping[str, Any], bool]:
        settings = methods.get(name, {})
        if not isinstance(settings, Mapping):
            raise TypeError(f"config.methods.{name} must be a mapping")
        unknown = sorted(set(settings) - allowed_settings[name])
        if unknown:
            raise ValueError(f"config.methods.{name} has unsupported keys: {unknown}")
        enabled = settings.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError(f"config.methods.{name}.enabled must be boolean")
        return settings, enabled

    m = len(steps)
    if m < 2:
        raise ValueError("offline averaging requires at least two checkpoints")
    distances = distances_from_latest(steps, interval)
    out: list[dict[str, Any]] = []

    _, lawa_enabled = method_settings("lawa")
    if lawa_enabled:
        out.append({"name": f"LAWA_M{m}", "method": "LAWA", "weights": lawa_weights(m)})

    _, wma_enabled = method_settings("wma")
    if wma_enabled:
        out.append({"name": f"WMA_M{m}", "method": "WMA", "weights": wma_weights(m)})

    lnwa_settings, lnwa_enabled = method_settings("lnwa")
    if lnwa_enabled:
        raw_penalties = lnwa_settings.get("lag_penalties")
        if not isinstance(raw_penalties, list) or not raw_penalties:
            raise ValueError("LNWA requires a non-empty lag_penalties list")
        penalties: list[float] = []
        for raw in raw_penalties:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("LNWA lag penalties must be numeric")
            penalty = float(raw)
            if not math.isfinite(penalty) or penalty < 0.0:
                raise ValueError("LNWA lag penalties must be finite and >= 0")
            if penalty in penalties:
                raise ValueError(f"duplicate LNWA lag penalty: {penalty}")
            penalties.append(penalty)
        for penalty in penalties:
            out.append(
                {
                    "name": f"LNWA_M{m}_lambda{_float_slug(penalty)}",
                    "method": "LNWA",
                    "weights": lnwa_weights(distances, penalty),
                    "lag_penalty": penalty,
                }
            )

    if not out:
        raise ValueError("at least one averaging method must be enabled")
    names = [item["name"] for item in out]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate endpoint names: {names}")
    for item in out:
        weights = [float(value) for value in item["weights"]]
        if len(weights) != m or not all(math.isfinite(value) for value in weights):
            raise RuntimeError(f"{item['name']} returned invalid checkpoint weights")
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"{item['name']} weights do not sum to one")
    return out


def run(config_path: Path) -> None:
    config = load_yaml(config_path)
    input_cfg = config.get("input")
    output_cfg = config.get("output")
    if not isinstance(input_cfg, Mapping) or not isinstance(output_cfg, Mapping):
        raise TypeError("config must contain input: and output: mappings")
    source_root, source_resolved = load_source_resolved(output_cfg)

    paths, steps, interval, retained_k = discover_snapshots(input_cfg)
    distances = distances_from_latest(steps, interval)
    latest_step = int(steps[-1])
    method_specs = build_method_weights(config, steps, interval)

    output_dir = Path(str(output_cfg["run_dir"])).expanduser().resolve()
    if output_dir == source_root:
        raise ValueError("output.run_dir must differ from output.source_run_dir")
    overwrite = bool(output_cfg.get("overwrite", False))
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output run directory is non-empty: {output_dir}; set output.overwrite: true or use a fresh path"
        )
    model, corpus, device, context = prepare_evaluator(config, source_resolved)
    output_dir.mkdir(parents=True, exist_ok=True)
    averaging_dir = output_dir / "branches" / "average"
    averaging_dir.mkdir(parents=True, exist_ok=True)

    family_counts: dict[str, int] = {}
    for spec in method_specs:
        family = str(spec["method"])
        family_counts[family] = family_counts.get(family, 0) + 1
    family_summary = ", ".join(f"{family}={count}" for family, count in family_counts.items())
    print(f"Loaded offline averaging config: {config_path.resolve()}", flush=True)
    print(
        f"Planned offline averaging endpoints: {len(method_specs)} ({family_summary})",
        flush=True,
    )

    resolved = copy.deepcopy(source_resolved)
    resolved["project"] = {
        "name": str(output_cfg.get("name", "offline_weighted_endpoints")),
        "output_dir": str(output_dir),
        "seed": int(config["runtime"].get("seed", 20260722)),
        "resume": False,
    }
    source_details = source_resolved.get("resolved", {})
    resolved["resolved"] = {"total_steps": latest_step}
    if "fork_step" in source_details:
        resolved["resolved"]["fork_step"] = source_details["fork_step"]
    resolved["postprocess"] = {
        "source_run": str(source_root),
        "checkpoint_steps": steps,
        "checkpoint_interval": interval,
        "methods": copy.deepcopy(config["methods"]),
        "evaluation": copy.deepcopy(config["evaluation"]),
    }
    atomic_json(output_dir / "resolved_config.json", resolved)
    # Mark a rerun incomplete before replacing any existing endpoints.
    atomic_json(output_dir / "final_endpoints_manifest.json", {"complete": False, "endpoints": []})

    declarations: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []

    for index, spec in enumerate(method_specs, start=1):
        name = str(spec["name"])
        weights = [float(value) for value in spec["weights"]]
        stats = weight_stats(weights, distances)
        print(
            f"[{index}/{len(method_specs)}] {name}: "
            f"m1={stats['first_temporal_moment']:.6g} "
            f"R={stats['noise_factor_R']:.6g} "
            f"neg_mass={stats['negative_weight_mass']:.6g}",
            flush=True,
        )
        state = linear_merge(paths, steps, weights)
        relative = f"branches/average/{name}.pt"
        method_metadata = {
            "name": name,
            "category": "averaging",
            "branch": "average",
            "step": latest_step,
            "method": spec["method"],
            "checkpoint_count": len(steps),
            "retained_checkpoint_capacity_K": retained_k,
            "selected_checkpoint_count_M": len(steps),
            "checkpoint_interval": interval,
            "checkpoint_steps_old_to_new": list(steps),
            "distances_from_latest_in_checkpoint_units": list(distances),
            "merge_kind": "global",
            "weights_old_to_new": weights,
            "weight_stats": stats,
        }
        if "lag_penalty" in spec:
            method_metadata["lag_penalty"] = spec["lag_penalty"]
        artifact = save_endpoint(output_dir / relative, state, method_metadata)
        evaluation_record = evaluate_endpoint_state(
            name=name,
            state=state,
            model=model,
            corpus=corpus,
            evaluation=config["evaluation"],
            device=device,
            context=context,
        )
        evaluation_record.update(
            {
                "method": spec["method"],
                "endpoint_artifact": {
                    "relative_path": relative,
                    **artifact,
                },
            }
        )
        evaluation_records.append(evaluation_record)
        atomic_json(output_dir / "evaluation" / "endpoints" / f"{name}.json", evaluation_record)
        declarations.append(
            {
                "name": name,
                "category": "averaging",
                "branch": "average",
                "relative_path": relative,
                "step": latest_step,
                **artifact,
                "metadata": method_metadata,
            }
        )
        del state
        gc.collect()

    final_manifest = {
        "complete": True,
        "total_steps": latest_step,
        "endpoints": declarations,
    }
    evaluation_summary = {
        "complete": True,
        "endpoint_count": len(evaluation_records),
        "results": evaluation_records,
    }
    atomic_json(output_dir / "evaluation" / "summary.json", evaluation_summary)
    atomic_json(output_dir / "final_endpoints_manifest.json", final_manifest)

    print(f"\nDone. PTQ endpoint directory: {output_dir}")
    print(f"Manifest: {output_dir / 'final_endpoints_manifest.json'}")
    print("Endpoints:")
    for item in declarations:
        print(f"  - {item['name']}: {item['relative_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate LAWA, WMA, and LNWA endpoints from Stable checkpoints."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
