from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .utils import read_json


def discover_sources(
    config: dict[str, Any], *, data_only: bool = False
) -> dict[str, Any]:
    source = config["source"]
    selection = source["selection"]
    categories = selection["include_categories"]
    branches = selection["include_branches"]
    excluded = set(source["exclude_endpoint_names"])
    endpoints, bindings, models = [], [], []
    for run in source["pretrain_runs"]:
        declared = run.get("categories", ["stable", "decay", "averaging"])
        if categories != "all" and not set(declared).intersection(categories):
            continue
        root = Path(run["path"]).expanduser().resolve()
        if data_only and not (root / "resolved_config.json").is_file():
            continue
        resolved = read_json(root / "resolved_config.json")
        model = resolved["model"]
        if model.get("architecture") != "opt":
            raise ValueError(f"only OPT sources are supported: {root}")
        models.append(model)
        bindings.append(_validation_binding(resolved))
        if data_only:
            continue
        manifest = read_json(root / "final_endpoints_manifest.json")
        if manifest.get("complete") is not True:
            raise ValueError(f"source endpoints are incomplete: {root}")
        entries = manifest.get("endpoints", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"source manifest has no endpoints: {root}")
        for item in entries:
            if item["category"] not in declared:
                continue
            if categories != "all" and item["category"] not in categories:
                continue
            if branches != "all" and item["branch"] not in branches:
                continue
            if item["name"] in excluded:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(item["name"])) or item[
                "name"
            ] in (".", ".."):
                raise ValueError("endpoint name must be a simple filename")
            relative = Path(item["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"endpoint path must be relative to its run: {relative}"
                )
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            entry = {key: item[key] for key in ("name", "category", "branch", "step")}
            entry.update(
                endpoint_key=f"{run['id']}::{item['name']}",
                run_id=run["id"],
                path=str(path),
                relative_path=str(relative),
                bytes=path.stat().st_size,
            )
            endpoints.append(entry)
    if not bindings:
        raise ValueError(
            "no configured source has resolved_config.json; run training first"
        )
    if any(model != models[0] for model in models[1:]):
        raise ValueError("PTQ sources must have the same OPT model configuration")
    if any(binding != bindings[0] for binding in bindings[1:]):
        raise ValueError("PTQ sources must use the same tokenizer and validation data")
    if not data_only and not endpoints:
        raise ValueError("source selection contains no endpoints")
    if len({item["endpoint_key"] for item in endpoints}) != len(endpoints):
        raise ValueError("source manifest contains duplicate endpoint names")
    binding = bindings[0]
    if config["calibration"]["sequence_length"] != binding["sequence_length"]:
        raise ValueError("calibration and source sequence lengths differ")
    return {"model": models[0], "validation": binding, "endpoints": endpoints}


def _validation_binding(resolved: dict[str, Any]) -> dict[str, Any]:
    data = resolved["data"]
    root = Path(data["root"]).expanduser().resolve()
    manifest = read_json(root / data.get("manifest", "tokenization_manifest.json"))
    identity = manifest.get("identity", {})
    path = root / data.get("validation_file", "validation_tokens.int32.bin")
    size = path.stat().st_size
    sequence_length = int(resolved["training"]["sequence_length"])
    blocks = int(resolved["evaluation"]["validation"]["num_blocks"])
    if (
        sequence_length < 2
        or blocks < 1
        or size < sequence_length * blocks * 4
        or size % 4
    ):
        raise ValueError(
            f"validation file does not contain the required int32 blocks: {path}"
        )
    if (
        "validation_file_bytes" in manifest
        and manifest["validation_file_bytes"] != size
    ):
        raise ValueError(f"validation file size differs from its manifest: {path}")
    dataset_name = identity.get("dataset_name", "HuggingFaceFW/fineweb-edu")
    if dataset_name != "HuggingFaceFW/fineweb-edu":
        raise ValueError("fineweb_edu_validation requires FineWeb-Edu source data")
    preparation = data.get("preparation", {})
    assets = preparation.get("tokenizer_assets_dir")
    if assets and Path(assets).expanduser().is_dir():
        assets = str(Path(assets).expanduser().resolve())
    elif (root / "tokenizer").is_dir():
        assets = str(root / "tokenizer")
    else:
        assets = None
    tokenizer = {
        key: identity.get(key)
        for key in (
            "tokenizer_name",
            "tokenizer_revision",
            "tokenizer_vocab_size",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        )
        if identity.get(key) is not None
    }
    tokenizer.setdefault("tokenizer_vocab_size", data.get("tokenizer_vocab_size"))
    if not assets and not tokenizer.get("tokenizer_name"):
        raise ValueError(
            "source data needs tokenizer_assets_dir or tokenizer_name metadata"
        )
    return {
        "path": str(path),
        "bytes": size,
        "sequence_length": sequence_length,
        "num_blocks": blocks,
        "dataset_name": dataset_name,
        "tokenizer": tokenizer,
        "tokenizer_assets_dir": assets,
    }


def load_endpoint_state(endpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(endpoint["path"], map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if (
        not isinstance(state, dict)
        or not state
        or not all(torch.is_tensor(v) for v in state.values())
    ):
        raise ValueError(f"endpoint lacks a model state dictionary: {endpoint['path']}")
    metadata = checkpoint.get("metadata", {})
    for key in ("name", "category", "branch", "step"):
        if key in metadata and metadata[key] != endpoint[key]:
            raise ValueError(
                f"endpoint {key} differs from its manifest: {endpoint['path']}"
            )
    return state
