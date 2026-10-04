from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def atomic_json(path: Path, payload: Any) -> None:
    """Durably publish JSON with atomic replacement and directory fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )


def safe_exp(value: float) -> float:
    return math.exp(value) if value < 80 else math.inf


def parameter_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    with torch.no_grad():
        for parameter in model.parameters():
            total.add_(parameter.detach().double().square().sum())
    return float(total.sqrt())
