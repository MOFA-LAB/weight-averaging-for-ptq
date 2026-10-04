from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, method: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PTQ config must be a YAML mapping")
    for section in (
        "project",
        "source",
        "calibration",
        "quantization",
        "evaluation",
        "runtime",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")
    project, source = config["project"], config["source"]
    if not project.get("output_dir"):
        raise ValueError("project.output_dir is required")
    project.setdefault("seed", 20260722)
    project.setdefault("resume", True)
    runs = source.get("pretrain_runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("source.pretrain_runs must contain at least one run")
    for run in runs:
        if not run.get("id") or not run.get("path"):
            raise ValueError("each source run requires id and path")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(run["id"])) or run["id"] in (
            ".",
            "..",
        ):
            raise ValueError("source run id must be a simple filename")
    if len({run["id"] for run in runs}) != len(runs):
        raise ValueError("source run ids must be unique")
    selection = source.setdefault("selection", {})
    selection.setdefault("include_categories", ["stable", "decay", "averaging"])
    selection.setdefault("include_branches", "all")
    for key in ("include_categories", "include_branches"):
        if selection[key] == ["all"]:
            selection[key] = "all"
        if selection[key] != "all" and (
            not isinstance(selection[key], list) or not selection[key]
        ):
            raise ValueError(f"source.selection.{key} must be all or a nonempty list")
    categories = selection["include_categories"]
    if categories != "all" and not set(categories) <= {"stable", "decay", "averaging"}:
        raise ValueError("unknown source category")
    source.setdefault("exclude_endpoint_names", [])
    calibration = config["calibration"]
    calibration.setdefault("text_field", "text")
    calibration.setdefault("micro_batch_size", 1)
    for key in ("cache_dir", "dataset_name", "dataset_config", "split", "revision"):
        if not calibration.get(key):
            raise ValueError(f"calibration.{key} is required")
    for key in (
        "num_sequences",
        "sequence_length",
        "shuffle_buffer_size",
        "micro_batch_size",
    ):
        _positive_int(calibration.get(key), f"calibration.{key}")
    if calibration["sequence_length"] < 2:
        raise ValueError("calibration.sequence_length must be at least two")
    quant = config["quantization"]
    bits = quant.get("bits")
    if (
        not isinstance(bits, list)
        or not bits
        or any(type(bit) is not int or bit not in (4, 3, 2) for bit in bits)
    ):
        raise ValueError("quantization.bits must be a nonempty subset of [4, 3, 2]")
    if len(bits) != len(set(bits)):
        raise ValueError("quantization.bits must not contain duplicates")
    quant["bits"] = [bit for bit in (4, 3, 2) if bit in bits]
    if quant.get("group_size") != 128:
        raise ValueError("the paper uses quantization.group_size: 128")
    if quant.get("artifact_mode") != "fake_dequant":
        raise ValueError("only fake_dequant artifacts are implemented")
    if method == "gptq":
        if quant.get("algorithm") != "gptq_second_order_error_compensation_v1":
            raise ValueError("unsupported GPTQ algorithm")
        if not 0 < float(quant.get("damp_percent", 0)) < 1:
            raise ValueError("quantization.damp_percent must be between zero and one")
        _positive_int(quant.get("column_block_size"), "quantization.column_block_size")
        if calibration["num_sequences"] % calibration["micro_batch_size"]:
            raise ValueError(
                "calibration.num_sequences must divide into micro_batch_size batches"
            )
    elif method == "awq":
        if quant.get("algorithm") != "awq_reference_v1":
            raise ValueError("unsupported AWQ algorithm")
        if quant.get("symmetric") is not False or quant.get("zero_point") is not True:
            raise ValueError("AWQ uses asymmetric quantization with a zero point")
        for key in ("alpha_grid", "clip_ratios"):
            values = quant.get(key)
            if (
                not isinstance(values, list)
                or not values
                or any(not 0 < float(v) <= 1 for v in values)
            ):
                raise ValueError(f"quantization.{key} must contain values in (0, 1]")
        for key in ("search_max_rows", "reconstruction_tokens", "row_chunk_size"):
            _positive_int(quant.get(key), f"quantization.{key}")
        quant.setdefault("exclude_modules", ["lm_head"])
    else:
        raise ValueError(f"unknown PTQ method: {method}")
    evaluation, runtime = config["evaluation"], config["runtime"]
    _positive_int(evaluation.get("micro_batch_size"), "evaluation.micro_batch_size")
    refined = evaluation.get("refinedweb_heldout", {})
    for key in (
        "dataset_name",
        "revision",
        "split",
        "text_field",
        "shard_path",
        "cache_dir",
    ):
        if not refined.get(key):
            raise ValueError(f"evaluation.refinedweb_heldout.{key} is required")
    if (
        Path(calibration["cache_dir"]).expanduser().resolve()
        == Path(refined["cache_dir"]).expanduser().resolve()
    ):
        raise ValueError(
            "calibration and evaluation require separate cache directories"
        )
    if runtime.get("device") not in ("cuda", "cpu"):
        raise ValueError("runtime.device must be cuda or cpu")
    if type(runtime.get("gpu_id", 0)) is not int or runtime.get("gpu_id", 0) < 0:
        raise ValueError("runtime.gpu_id must be a nonnegative integer")
    precision = evaluation.get("precision", runtime.get("precision", "bf16"))
    if precision not in ("bf16", "bf16_mixed", "fp32"):
        raise ValueError("precision must be bf16, bf16_mixed or fp32")
    return config


def resume_config(config: dict[str, Any]) -> dict[str, Any]:
    """Settings that must agree when continuing results in the same directory."""
    result = copy.deepcopy(config)
    for key in ("resume", "output_dir", "name"):
        result["project"].pop(key, None)
    result["runtime"].pop("gpu_id", None)
    return result


def _positive_int(value: Any, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
