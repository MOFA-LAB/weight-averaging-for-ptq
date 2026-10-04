# Rank-local I/O errors are collected before continuing distributed execution.
# ruff: noqa: BLE001
from __future__ import annotations

import json
import random
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.artifacts import durable_torch_save
from src.distributed import DistributedContext

FORMAT_VERSION = 3


class ArtifactError(RuntimeError):
    pass


def _alias_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        str(tensor.device),
        str(tensor.dtype),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def synchronize_errors(
    context: DistributedContext, local_error: str | None, operation: str
) -> None:
    errors = context.gather_objects(local_error)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise ArtifactError(f"{operation} failed; " + " | ".join(failures))


def cpu_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Copy parameters to CPU, retaining tied embedding/output weights."""
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    copies: dict[tuple[Any, ...], torch.Tensor] = {}
    for name, value in model.state_dict().items():
        key = _alias_key(value)
        if key not in copies:
            copies[key] = value.detach().to(device="cpu", copy=True)
        result[name] = copies[key]
    return result


def rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device).cpu()
        if device.type == "cuda"
        else None,
    }


def restore_rng_state(payload: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(payload["torch_cuda"], device)


def save_recovery_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    branch: str,
    device: torch.device,
    context: DistributedContext,
) -> None:
    states = context.gather_objects(rng_state(device))
    error = None
    if context.is_main:
        try:
            durable_torch_save(
                path,
                {
                    "format_version": FORMAT_VERSION,
                    "step": int(step),
                    "branch": branch,
                    "model": cpu_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "rng_states": states,
                },
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    synchronize_errors(context, error, f"checkpoint save {path}")


def load_recovery_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_branch: str,
    device: torch.device,
    context: DistributedContext,
) -> int:
    step = -1
    error = None
    try:
        # Recovery files contain Python/NumPy RNG states. Only load checkpoints
        # produced by your own run; parameter-only published files use weights_only.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != FORMAT_VERSION:
            raise ArtifactError(f"unsupported recovery checkpoint: {path}")
        if payload.get("branch") != expected_branch:
            raise ArtifactError(f"checkpoint branch differs: {path}")
        step = int(payload["step"])
        if step < 0:
            raise ArtifactError(f"checkpoint step is negative: {path}")
        if len(payload["rng_states"]) != context.world_size:
            raise ArtifactError("checkpoint world size differs from this launch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng_states"][context.rank], device)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    synchronize_errors(context, error, f"checkpoint load {path}")
    return step


def save_endpoint(
    path: Path,
    *,
    state: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    context: DistributedContext,
) -> dict[str, Any]:
    result = None
    error = None
    if context.is_main:
        try:
            result = durable_torch_save(
                path,
                {
                    "format_version": FORMAT_VERSION,
                    "metadata": dict(metadata),
                    "model": OrderedDict(state),
                },
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    synchronize_errors(context, error, f"endpoint save {path}")
    return context.broadcast_from_main(result)


def load_endpoint(
    path: Path,
    *,
    model: nn.Module,
    expected_metadata: Mapping[str, Any],
    context: DistributedContext,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    error = None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = dict(payload["metadata"])
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ArtifactError(f"endpoint {key} differs: {path}")
        model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    synchronize_errors(context, error, f"endpoint load {path}")
    return metadata


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"JSON artifact must be a mapping: {path}")
    return payload
