"""Train a common OPT trunk, then Stable and cosine-decay continuations."""

# Rank-local exceptions must reach every DDP worker before any worker proceeds.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import copy
import fcntl
import math
import shutil
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.artifacts import durable_torch_save
from src.checkpoint_snapshots import BranchSnapshotStore
from src.config import load_config, public_config, resolve_path
from src.data import TokenCorpus, load_corpus
from src.distributed import (
    DistributedContext,
    initialize_distributed,
    shutdown_distributed,
)
from src.experiment_plan import (
    PhasePlan,
    decay_learning_rate_at_step,
    resolve_decay_branch_plans,
    resolve_phase_plan,
    trajectory_steps,
    trunk_learning_rate,
)
from src.hessian import estimate_curvature
from src.model import build_adamw, build_model
from src.report import write_report
from src.unified_artifacts import (
    cpu_state_dict,
    load_endpoint,
    load_recovery_checkpoint,
    read_json,
    restore_rng_state,
    rng_state,
    save_endpoint,
    save_recovery_checkpoint,
    synchronize_errors,
)
from src.utils import (
    append_jsonl,
    atomic_json,
    parameter_norm,
    safe_exp,
    seed_everything,
)


def effective_train_blocks(config: Mapping[str, Any], corpus: TokenCorpus) -> int:
    budget = config["training"].get("train_token_budget")
    requested = (
        corpus.train_blocks if budget is None else int(budget) // corpus.sequence_length
    )
    if requested > corpus.train_blocks:
        raise ValueError("training.train_token_budget exceeds the prepared token cache")
    return requested


def _global_batch_sequences(config: Mapping[str, Any], world_size: int) -> int:
    training = config["training"]
    return (
        int(training["micro_batch_size_per_gpu"])
        * world_size
        * int(training["gradient_accumulation_steps"])
    )


def _step_local_batch_layout(
    *,
    plan: PhasePlan,
    step: int,
    world_size: int,
    rank: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[tuple[int, int], ...]:
    """Disjoint rank microbatches, including the exact final partial update."""
    if not 0 <= rank < world_size or not 1 <= step <= plan.total_steps:
        raise ValueError("invalid rank or optimizer step")
    if (
        world_size * micro_batch_size * gradient_accumulation_steps
        != plan.global_batch_size
    ):
        raise ValueError("runtime batch size differs from the phase plan")
    start = (step - 1) * plan.global_batch_size
    count = min(plan.global_batch_size, plan.train_blocks - start)
    chunks = []
    consumed = 0
    while consumed < count:
        round_count = min(world_size * micro_batch_size, count - consumed)
        base, remainder = divmod(round_count, world_size)
        size = base + int(rank < remainder)
        chunks.append((start + consumed + rank * base + min(rank, remainder), size))
        consumed += round_count
    return tuple(chunks)


def _autocast(precision: str, device: torch.device):
    if precision == "bf16_mixed":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"unsupported precision: {precision}")


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    corpus: TokenCorpus,
    protocol: Mapping[str, Any],
    device: torch.device,
    context: DistributedContext,
) -> dict[str, float | int]:
    count = int(protocol["num_blocks"])
    micro = int(protocol["micro_batch_size"])
    if count > corpus.validation_blocks:
        raise ValueError(
            f"validation needs {count} blocks, cache has {corpus.validation_blocks}"
        )
    flags = {module: module.training for module in model.modules()}
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    error = None
    try:
        model.eval()
        for batch_index in range(
            context.rank, math.ceil(count / micro), context.world_size
        ):
            start = batch_index * micro
            input_ids = corpus.batch(
                "validation", start, min(micro, count - start), device
            )
            with _autocast(str(protocol["precision"]), device):
                logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
                targets = input_ids[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
            totals[0] += loss.double()
            totals[1] += targets.numel()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for module, flag in flags.items():
            module.training = flag
    synchronize_errors(context, error, "validation")
    context.all_reduce_sum(totals)
    loss = float(totals[0] / totals[1])
    if not math.isfinite(loss):
        raise RuntimeError("non-finite validation loss")
    return {
        "validation_loss": loss,
        "validation_perplexity": safe_exp(loss),
        "validation_tokens": int(totals[1]),
        "validation_blocks": count,
        "validation_batches": math.ceil(count / micro),
    }


def _evaluate_curvature_if_enabled(
    model: nn.Module,
    corpus: TokenCorpus,
    protocol: Mapping[str, Any],
    device: torch.device,
    context: DistributedContext,
) -> dict[str, Any]:
    if not protocol["enabled"]:
        return {}
    return estimate_curvature(
        model,
        corpus,
        {key: value for key, value in protocol.items() if key != "enabled"},
        device,
        context,
    )


def _train_optimizer_step(
    *,
    ddp_model: nn.Module,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    corpus: TokenCorpus,
    config: Mapping[str, Any],
    plan: PhasePlan,
    step: int,
    device: torch.device,
    context: DistributedContext,
) -> tuple[float, float]:
    training = config["training"]
    layout = _step_local_batch_layout(
        plan=plan,
        step=step,
        world_size=context.world_size,
        rank=context.rank,
        micro_batch_size=int(training["micro_batch_size_per_gpu"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
    )
    global_count = min(
        plan.global_batch_size, plan.train_blocks - (step - 1) * plan.global_batch_size
    )
    raw_model.train()
    optimizer.zero_grad(set_to_none=True)
    local_loss = torch.zeros((), dtype=torch.float64, device=device)
    for index, (start, size) in enumerate(layout):
        sync = (
            ddp_model.no_sync()
            if isinstance(ddp_model, DDP) and index + 1 < len(layout)
            else nullcontext()
        )
        with sync:
            # If the final microbatch has fewer sequences than ranks, empty ranks
            # still participate in DDP with a zero-weight dummy forward.
            input_ids = corpus.batch(
                "train", start if size else 0, max(size, 1), device
            )
            with _autocast(str(training["precision"]), device):
                logits = ddp_model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1)
                )
            if not context.all_true(bool(torch.isfinite(loss).item()), device):
                raise RuntimeError(f"non-finite loss at step {step}")
            # DDP averages gradients over ranks, so compensate when final rank
            # batches have unequal lengths. Each real sequence has weight 1/N.
            weight = size * context.world_size / global_count
            (loss * weight).backward()
        local_loss += loss.detach().double() * weight
    grad_norm = torch.nn.utils.clip_grad_norm_(
        raw_model.parameters(), float(training["max_grad_norm"])
    )
    if not context.all_true(bool(torch.isfinite(grad_norm).item()), device):
        raise RuntimeError(f"non-finite gradient at step {step}")
    optimizer.step()
    context.all_reduce_sum(local_loss)
    return float(local_loss / context.world_size), float(grad_norm)


def _write_main(context: DistributedContext, path: Path, payload: Any) -> None:
    error = None
    if context.is_main:
        try:
            atomic_json(path, payload)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    synchronize_errors(context, error, f"write {path}")


def _training_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Directly comparable settings needed to resume or reuse a common trunk."""
    return {
        "seed": config["project"]["seed"],
        "model": config["model"],
        "training": config["training"],
        "optimizer": config["optimizer"],
        "data": {
            key: config["data"][key]
            for key in (
                "root",
                "manifest",
                "train_file",
                "validation_file",
                "storage_dtype",
                "tokenizer_vocab_size",
            )
        },
    }


def _setup_run(
    config: dict[str, Any],
    run_dir: Path,
    plan: PhasePlan,
    corpus: TokenCorpus,
    context: DistributedContext,
):
    handle = None
    error = None
    if context.is_main:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            handle = (run_dir / ".run.lock").open("a+")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = run_dir / "resolved_config.json"
            occupied = any(item.name != ".run.lock" for item in run_dir.iterdir())
            if occupied and (not path.is_file() or not config["project"]["resume"]):
                raise RuntimeError(
                    "output directory is not empty; use a fresh output_dir or enable resume"
                )
            resolved = public_config(config)
            resolved["resolved"] = {
                **asdict(plan),
                "world_size": context.world_size,
                "global_batch_sequences": plan.global_batch_size,
                "effective_train_tokens": plan.train_blocks * corpus.sequence_length,
            }
            resolved["data_summary"] = {
                key: corpus.manifest[key]
                for key in ("train_file_bytes", "validation_file_bytes")
            }
            if path.is_file():
                previous = read_json(path)
                if _training_settings(previous) != _training_settings(resolved):
                    raise RuntimeError(
                        "training settings changed; use a fresh output_dir"
                    )
                for key in (
                    "post_fork",
                    "evaluation",
                    "initialization",
                    "resolved",
                    "data_summary",
                ):
                    if previous.get(key) != resolved.get(key):
                        raise RuntimeError(f"{key} changed; use a fresh output_dir")
            atomic_json(path, resolved)
        except Exception as exc:
            if handle:
                handle.close()
                handle = None
            error = f"{type(exc).__name__}: {exc}"
    synchronize_errors(context, error, "run setup")
    return handle


def _measure(
    *,
    model: nn.Module,
    corpus: TokenCorpus,
    config: Mapping[str, Any],
    branch: str,
    step: int,
    path: Path,
    final: bool,
    device: torch.device,
    context: DistributedContext,
) -> dict[str, Any]:
    cached = None
    if context.is_main and path.is_file():
        try:
            cached = read_json(path)
        except (OSError, ValueError):
            pass
    cached = context.broadcast_from_main(cached)
    if (
        cached is not None
        and cached.get("step") == step
        and cached.get("branch") == branch
        and "validation_loss" in cached
        and "validation_perplexity" in cached
    ):
        return cached
    state = rng_state(device)
    try:
        curvature = config["evaluation"]["final" if final else "trajectory"][
            "curvature"
        ]
        record = {
            "branch": branch,
            "step": step,
            "time": time.time(),
            "curvature_enabled": bool(curvature["enabled"]),
            **evaluate_validation(
                model, corpus, config["evaluation"]["validation"], device, context
            ),
            **_evaluate_curvature_if_enabled(model, corpus, curvature, device, context),
        }
        if final:
            record["parameter_norm"] = parameter_norm(model)
    finally:
        restore_rng_state(state, device)
    _write_main(context, path, record)
    if context.is_main:
        print(
            f"[evaluate] {branch} step={step} ppl={record['validation_perplexity']:.4f}",
            flush=True,
        )
    return record


def _import_fork(
    *,
    config: dict[str, Any],
    run_dir: Path,
    plan: PhasePlan,
    stores: Mapping[str, BranchSnapshotStore],
    context: DistributedContext,
) -> None:
    if context.broadcast_from_main(
        (run_dir / "checkpoints" / "fork.pt").is_file() if context.is_main else None
    ):
        return
    error = None
    if context.is_main:
        try:
            target = run_dir / "checkpoints" / "fork.pt"
            source = resolve_path(config, config["initialization"]["source_run"])
            previous = read_json(source / "resolved_config.json")
            if _training_settings(previous) != _training_settings(config):
                raise RuntimeError(
                    "source run has different model, data, optimizer or training settings"
                )
            if previous["resolved"]["fork_step"] != plan.fork_step:
                raise RuntimeError("source run has a different fork step")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Copy historical common-trunk snapshots before publishing the fork.
            # An imported run must not silently average fewer than K checkpoints.
            for store in stores.values():
                for step in store.required_steps:
                    if step > plan.fork_step:
                        continue
                    candidates = [
                        source / "stable" / f"step_{step:08d}.pt",
                        *sorted((source / "decay").glob(f"*/step_{step:08d}.pt")),
                    ]
                    snapshot = next(
                        (item for item in candidates if item.is_file()), None
                    )
                    if snapshot is None:
                        raise FileNotFoundError(
                            f"source run lacks common-trunk snapshot W_{step}"
                        )
                    payload = torch.load(
                        snapshot, map_location="cpu", weights_only=True
                    )
                    if payload["step"] != step:
                        raise ValueError(f"snapshot step mismatch: {snapshot}")
                    payload["branch"] = store.branch
                    durable_torch_save(store.path(step), payload)
            temporary = target.with_suffix(".tmp")
            shutil.copyfile(source / "checkpoints" / "fork.pt", temporary)
            temporary.replace(target)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    synchronize_errors(context, error, "fork import")


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    context = initialize_distributed(
        int(config["training"]["expected_world_size"]), config["runtime"]["device"]
    )
    device = (
        torch.device("cuda", context.local_rank)
        if config["runtime"]["device"] == "cuda"
        else torch.device("cpu")
    )
    lock = None
    try:
        if (
            device.type == "cuda"
            and config["training"]["precision"] == "bf16_mixed"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("bf16_mixed requires a BF16-capable GPU")
        seed_everything(
            int(config["project"]["seed"]),
            bool(config["training"]["deterministic_algorithms"]),
        )
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(config["training"]["tf32"])
        data_config = copy.deepcopy(config["data"])
        data_config["root"] = str(resolve_path(config, data_config["root"]))
        corpus = None
        error = None
        try:
            corpus = load_corpus(
                data_config, int(config["training"]["sequence_length"])
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        synchronize_errors(context, error, "load token cache")
        blocks = effective_train_blocks(config, corpus)
        plan = resolve_phase_plan(
            blocks,
            _global_batch_sequences(config, context.world_size),
            config["training"]["warmup_ratio"],
            config["training"]["stable_ratio"],
        )
        branches = list(config["post_fork"]["selected_branches"])
        decay_plans = {
            item.name: item
            for item in resolve_decay_branch_plans(
                config["post_fork"]["decay"]["branches"],
                plan.total_steps,
                plan.fork_step,
            )
        }
        run_dir = resolve_path(config, config["project"]["output_dir"])
        lock = _setup_run(config, run_dir, plan, corpus, context)
        stores = {
            branch: BranchSnapshotStore(
                run_dir=run_dir,
                branch=branch,
                total_steps=plan.total_steps,
                **config["post_fork"]["checkpoint_snapshots"],
            )
            for branch in branches
        }
        if config["initialization"]["mode"] == "fork":
            _import_fork(
                config=config,
                run_dir=run_dir,
                plan=plan,
                stores=stores,
                context=context,
            )
        model = build_model(config["model"], int(config["project"]["seed"])).to(device)
        ddp_model = (
            DDP(
                model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            if context.enabled
            else model
        )
        # Identical initial weights with independent dropout streams per rank.
        seed_everything(
            int(config["project"]["seed"]) + context.rank,
            bool(config["training"]["deterministic_algorithms"]),
        )
        optimizer = build_adamw(model, config["optimizer"])
        peak_lr = float(config["optimizer"]["learning_rate"])
        fork_path = run_dir / "checkpoints" / "fork.pt"
        latest_trunk = run_dir / "checkpoints" / "latest.pt"
        trajectory_due = set(
            trajectory_steps(
                plan,
                **{
                    key: config["evaluation"]["trajectory"][key]
                    for key in ("interval", "include_fork", "include_final")
                },
            )
        )

        def load_checkpoint(path: Path, branch: str) -> int:
            return load_recovery_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                expected_branch=branch,
                device=device,
                context=context,
            )

        def save_checkpoint(path: Path, branch: str, step: int) -> None:
            save_recovery_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=step,
                branch=branch,
                device=device,
                context=context,
            )

        def update(
            step: int, branch: str, lr: float, active_stores: list[BranchSnapshotStore]
        ) -> None:
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, norm = _train_optimizer_step(
                ddp_model=ddp_model,
                raw_model=model,
                optimizer=optimizer,
                corpus=corpus,
                config=config,
                plan=plan,
                step=step,
                device=device,
                context=context,
            )
            if context.is_main and (
                step % config["training"]["log_interval"] == 0
                or step == plan.total_steps
            ):
                record = {
                    "branch": branch,
                    "step": step,
                    "loss": loss,
                    "learning_rate": lr,
                    "gradient_norm": norm,
                }
                append_jsonl(run_dir / "branches" / branch / "train.jsonl", record)
                print(
                    f"[train] {branch} {step}/{plan.total_steps} loss={loss:.6f} lr={lr:.6g}",
                    flush=True,
                )
            for store in active_stores:
                store.save_if_due(model=model, step=step, context=context)
            if step in trajectory_due:
                optimizer.zero_grad(set_to_none=True)
                measured_branch = (
                    "fork"
                    if step == plan.fork_step
                    else "stable"
                    if branch == "trunk"
                    else branch
                )
                _measure(
                    model=model,
                    corpus=corpus,
                    config=config,
                    branch=measured_branch,
                    step=step,
                    path=run_dir
                    / "trajectory"
                    / measured_branch
                    / f"step_{step:08d}.json",
                    final=False,
                    device=device,
                    context=context,
                )

        if fork_path.is_file():
            current = load_checkpoint(fork_path, "trunk")
            if current != plan.fork_step:
                raise ValueError(
                    "fork checkpoint step differs from configured phase plan"
                )
        else:
            current = (
                load_checkpoint(latest_trunk, "trunk") if latest_trunk.is_file() else 0
            )
            if not 0 <= current <= plan.fork_step:
                raise ValueError("trunk recovery step is outside the common trunk")
            for store in stores.values():
                store.ensure_through(current_step=current, model=model, context=context)
            for step in range(current + 1, plan.fork_step + 1):
                update(
                    step,
                    "trunk",
                    trunk_learning_rate(step, plan, peak_lr),
                    list(stores.values()),
                )
                if step % config["training"]["recovery_checkpoint_interval"] == 0:
                    save_checkpoint(latest_trunk, "trunk", step)
            save_checkpoint(fork_path, "trunk", plan.fork_step)
        for store in stores.values():
            store.ensure_through(
                current_step=plan.fork_step, model=model, context=context
            )

        endpoints = []
        for branch in branches:
            optimizer = build_adamw(model, config["optimizer"])
            branch_dir = run_dir / "branches" / branch
            final_checkpoint = branch_dir / "checkpoints" / "final.pt"
            latest = branch_dir / "checkpoints" / "latest.pt"
            if final_checkpoint.is_file():
                current = load_checkpoint(final_checkpoint, branch)
            elif latest.is_file():
                current = load_checkpoint(latest, branch)
            else:
                # Restore parameters, AdamW moments and per-rank RNG before EACH
                # branch; no continuation inherits another branch's training.
                current = load_checkpoint(fork_path, "trunk")
            if not plan.fork_step <= current <= plan.total_steps:
                raise ValueError(f"{branch} recovery step is outside the continuation")
            stores[branch].ensure_through(
                current_step=current, model=model, context=context
            )
            for step in range(current + 1, plan.total_steps + 1):
                lr = (
                    peak_lr
                    if branch == "stable"
                    else decay_learning_rate_at_step(
                        step, plan=decay_plans[branch], peak_lr=peak_lr
                    )
                )
                update(step, branch, lr, [stores[branch]])
                if step % config["training"]["recovery_checkpoint_interval"] == 0:
                    save_checkpoint(latest, branch, step)
            save_checkpoint(final_checkpoint, branch, plan.total_steps)
            metadata = {
                "name": "stable_W" if branch == "stable" else f"{branch}_W",
                "category": "stable" if branch == "stable" else "decay",
                "branch": branch,
                "step": plan.total_steps,
            }
            endpoint = branch_dir / "final_W_state.pt"
            save_endpoint(
                endpoint,
                state=cpu_state_dict(model) if context.is_main else {},
                metadata=metadata,
                context=context,
            )
            stores[branch].finalize(context)
            endpoints.append(
                {
                    **metadata,
                    "relative_path": endpoint.relative_to(run_dir).as_posix(),
                    "metadata": metadata,
                }
            )
        _write_main(
            context,
            run_dir / "final_endpoints_manifest.json",
            {
                "complete": True,
                "total_steps": plan.total_steps,
                "endpoint_count": len(endpoints),
                "endpoints": endpoints,
            },
        )
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        ddp_model = model
        final_records = []
        for endpoint in endpoints:
            load_endpoint(
                run_dir / endpoint["relative_path"],
                model=model,
                expected_metadata=endpoint["metadata"],
                context=context,
            )
            record = _measure(
                model=model,
                corpus=corpus,
                config=config,
                branch=endpoint["branch"],
                step=plan.total_steps,
                path=run_dir
                / config["evaluation"]["final"]["output_directory"]
                / f"{endpoint['name']}.json",
                final=True,
                device=device,
                context=context,
            )
            final_records.append({**record, **endpoint["metadata"]})
        _write_main(
            context,
            run_dir
            / config["evaluation"]["final"]["output_directory"]
            / "summary.json",
            {"complete": True, "results": final_records},
        )
        error = None
        if context.is_main:
            try:
                trajectories = [
                    read_json(path)
                    for path in sorted((run_dir / "trajectory").glob("*/step_*.json"))
                ]
                write_report(
                    run_dir / "report",
                    final_records,
                    trajectories,
                    branch_order=branches,
                    fork_step=plan.fork_step,
                )
                atomic_json(
                    run_dir / "COMPLETED",
                    {
                        "complete": True,
                        "total_steps": plan.total_steps,
                        "endpoints": [item["name"] for item in endpoints],
                    },
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronize_errors(context, error, "final report")
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        shutdown_distributed(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run(parser.parse_args().config)
