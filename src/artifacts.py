from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_torch_save(path: Path, payload: Any) -> dict[str, Any]:
    """Atomically save a torch payload and fsync both file and directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        size = temporary.stat().st_size
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return {"bytes": int(size)}
