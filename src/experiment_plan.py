from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhasePlan:
    train_blocks: int
    global_batch_size: int
    total_steps: int
    full_batch_steps: int
    final_step_batch_size: int
    warmup_steps: int
    pre_fork_stable_steps: int
    tail_steps: int
    fork_step: int


@dataclass(frozen=True)
class DecayBranchPlan:
    name: str
    schedule: str
    stable_ratio_after_fork: float
    decay_ratio: float
    final_lr_ratio: float
    fork_step: int
    total_steps: int
    stable_steps: int
    decay_steps: int
    stable_end_step: int
    first_decay_step: int


def resolve_phase_plan(
    train_blocks: int,
    global_batch_size: int,
    warmup_ratio: float,
    stable_ratio: float,
) -> PhasePlan:
    blocks = int(train_blocks)
    batch = int(global_batch_size)
    if blocks < 1 or batch < 1:
        raise ValueError("train_blocks and global_batch_size must be positive")
    # Preserve the configured token prefix exactly.  A token budget normally is
    # not divisible by the global batch, so the last optimizer update is a
    # smaller (but still distributed) batch instead of silently discarding the
    # remainder of the prefix.
    full_batch_steps, remainder = divmod(blocks, batch)
    total = full_batch_steps + int(remainder > 0)
    if total < 3:
        raise ValueError(
            "the training stream must contain at least three optimizer updates"
        )
    warmup = round(total * float(warmup_ratio))
    stable = round(total * float(stable_ratio))
    tail = total - warmup - stable
    if min(warmup, stable, tail) < 1:
        raise ValueError(
            "resolved warmup, pre-fork Stable, and suffix must be non-empty"
        )
    return PhasePlan(
        train_blocks=blocks,
        global_batch_size=batch,
        total_steps=total,
        full_batch_steps=full_batch_steps,
        final_step_batch_size=remainder or batch,
        warmup_steps=warmup,
        pre_fork_stable_steps=stable,
        tail_steps=tail,
        fork_step=warmup + stable,
    )


def trunk_learning_rate(step: int, plan: PhasePlan, peak_lr: float) -> float:
    current = int(step)
    if not 1 <= current <= plan.fork_step:
        raise ValueError(f"trunk step must lie in [1,{plan.fork_step}]")
    peak = float(peak_lr)
    if current <= plan.warmup_steps:
        return peak * current / plan.warmup_steps
    return peak


def resolve_decay_branch_plans(
    branches: Sequence[Mapping[str, Any]],
    total_steps: int,
    fork_step: int,
) -> tuple[DecayBranchPlan, ...]:
    total = int(total_steps)
    fork = int(fork_step)
    if not 0 <= fork < total:
        raise ValueError("expected 0 <= fork_step < total_steps")
    tail = total - fork
    if not branches:
        raise ValueError("at least one decay branch is required")
    result: list[DecayBranchPlan] = []
    names: set[str] = set()
    horizon: float | None = None
    for index, raw in enumerate(branches):
        spec = dict(raw)
        name = str(spec.get("name", ""))
        if not name or name in names:
            raise ValueError(
                f"invalid or duplicate decay branch name at index {index}: {name}"
            )
        names.add(name)
        if spec.get("schedule") != "stable_then_cosine_decay":
            raise ValueError(f"branch {name} has an unsupported schedule")
        stable_ratio = _finite(
            spec.get("stable_ratio_after_fork"), f"{name}.stable_ratio"
        )
        decay_ratio = _finite(spec.get("decay_ratio"), f"{name}.decay_ratio")
        final_ratio = _finite(spec.get("final_lr_ratio"), f"{name}.final_lr_ratio")
        if stable_ratio < 0.0 or decay_ratio <= 0.0:
            raise ValueError(f"branch {name} has an invalid continuation ratio")
        if not 0.0 <= final_ratio <= 1.0:
            raise ValueError(f"branch {name}.final_lr_ratio must lie in [0,1]")
        branch_horizon = stable_ratio + decay_ratio
        if horizon is None:
            horizon = branch_horizon
        elif not math.isclose(branch_horizon, horizon, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("all decay branches must cover the same suffix")
        if abs(round(total * branch_horizon) - tail) > 1:
            raise ValueError(f"branch {name} ratios do not match the resolved suffix")
        stable_steps = round(total * stable_ratio)
        decay_steps = tail - stable_steps
        if stable_steps < 0 or decay_steps < 1:
            raise ValueError(f"branch {name} must leave a non-empty decay")
        if abs(decay_steps - round(total * decay_ratio)) > 1:
            raise ValueError(
                f"branch {name} decay ratio does not match the resolved suffix"
            )
        stable_end = fork + stable_steps
        result.append(
            DecayBranchPlan(
                name=name,
                schedule="stable_then_cosine_decay",
                stable_ratio_after_fork=stable_ratio,
                decay_ratio=decay_ratio,
                final_lr_ratio=final_ratio,
                fork_step=fork,
                total_steps=total,
                stable_steps=stable_steps,
                decay_steps=decay_steps,
                stable_end_step=stable_end,
                first_decay_step=stable_end + 1,
            )
        )
    return tuple(result)


def decay_learning_rate_at_step(
    step: int,
    *,
    plan: DecayBranchPlan,
    peak_lr: float,
) -> float:
    current = int(step)
    if not plan.fork_step <= current <= plan.total_steps:
        raise ValueError(
            f"decay step must lie in [{plan.fork_step},{plan.total_steps}]"
        )
    peak = float(peak_lr)
    if current <= plan.stable_end_step:
        return peak
    progress = (current - plan.stable_end_step) / plan.decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (plan.final_lr_ratio + (1.0 - plan.final_lr_ratio) * cosine)


def trajectory_steps(
    plan: PhasePlan,
    *,
    interval: int,
    include_fork: bool,
    include_final: bool,
) -> tuple[int, ...]:
    spacing = int(interval)
    if spacing < 1:
        raise ValueError("trajectory interval must be positive")
    # Use an absolute optimizer-step grid so independently configured runs and
    # all post-fork branches share the same x-axis.  The first measurement is
    # after ``interval`` completed updates, never at untrained W_0.
    steps: set[int] = set(range(spacing, plan.total_steps, spacing))
    if include_fork:
        steps.add(plan.fork_step)
    if include_final:
        steps.add(plan.total_steps)
    return tuple(sorted(steps))


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
