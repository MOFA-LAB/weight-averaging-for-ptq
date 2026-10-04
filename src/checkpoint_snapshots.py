# Publish rank-zero I/O failures to every DDP worker.
# ruff: noqa: BLE001
from __future__ import annotations

from pathlib import Path

from torch import nn

from src.artifacts import durable_torch_save
from src.distributed import DistributedContext
from src.unified_artifacts import cpu_state_dict, synchronize_errors
from src.utils import atomic_json


def retained_snapshot_steps(
    total_steps: int, interval: int, keep_last: int
) -> tuple[int, ...]:
    """K final-aligned checkpoints span K-1 intervals, including before the fork."""
    if min(total_steps, interval, keep_last) < 1:
        raise ValueError(
            "snapshot total_steps, interval and keep_last must be positive"
        )
    return tuple(
        sorted(
            step for i in range(keep_last) if (step := total_steps - i * interval) > 0
        )
    )


def snapshot_directory(run_dir: Path, branch: str) -> Path:
    return run_dir / "stable" if branch == "stable" else run_dir / "decay" / branch


class BranchSnapshotStore:
    def __init__(
        self,
        *,
        run_dir: Path,
        branch: str,
        total_steps: int,
        interval: int,
        keep_last: int,
        available_start_step: int = 1,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.branch = branch
        self.root = snapshot_directory(self.run_dir, branch)
        self.total_steps = total_steps
        self.interval = interval
        self.keep_last = keep_last
        self.required_steps = tuple(
            step
            for step in retained_snapshot_steps(total_steps, interval, keep_last)
            if step >= available_start_step
        )

    def path(self, step: int) -> Path:
        return self.root / f"step_{step:08d}.pt"

    def save_if_due(
        self, *, model: nn.Module, step: int, context: DistributedContext
    ) -> None:
        if step not in self.required_steps:
            return
        error = None
        if context.is_main:
            try:
                durable_torch_save(
                    self.path(step),
                    {
                        "model": cpu_state_dict(model),
                        "branch": self.branch,
                        "step": step,
                    },
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronize_errors(context, error, f"snapshot save {self.branch} W_{step}")

    def ensure_through(
        self, *, current_step: int, model: nn.Module, context: DistributedContext
    ) -> None:
        error = None
        if context.is_main:
            missing = [
                step
                for step in self.required_steps
                if step < current_step and not self.path(step).is_file()
            ]
            if missing:
                error = f"cannot reconstruct missing historical snapshots: {missing}"
        synchronize_errors(context, error, f"snapshot resume {self.branch}")
        self.save_if_due(model=model, step=current_step, context=context)

    def finalize(self, context: DistributedContext) -> None:
        error = None
        if context.is_main:
            try:
                for step in self.required_steps:
                    if not self.path(step).is_file():
                        raise FileNotFoundError(self.path(step))
                atomic_json(
                    self.root / "manifest.json",
                    {
                        "complete": True,
                        "branch": self.branch,
                        "total_steps": self.total_steps,
                        "interval": self.interval,
                        "keep_last": self.keep_last,
                        "steps": list(self.required_steps),
                        "snapshots": [
                            {
                                "step": step,
                                "relative_path": self.path(step)
                                .relative_to(self.run_dir)
                                .as_posix(),
                            }
                            for step in self.required_steps
                        ],
                    },
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronize_errors(context, error, f"snapshot manifest {self.branch}")
