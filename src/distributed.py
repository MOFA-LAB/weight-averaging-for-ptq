from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

PROCESS_GROUP_TIMEOUT = timedelta(minutes=10)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def all_true(self, value: bool, device: torch.device) -> bool:
        tensor = torch.tensor(int(value), dtype=torch.int32, device=device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(tensor.item())

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        result: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(result, value)
        return result

    def broadcast_from_main(self, value: Any) -> Any:
        if not self.enabled:
            return value
        payload = [value if self.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]


def initialize_distributed(
    expected_world_size: int, device_type: str = "cuda"
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(expected_world_size):
        raise RuntimeError(
            f"expected exactly {expected_world_size} GPU workers, got {world_size}; "
            "set runtime.gpu_ids in the experiment config and launch with launch.py"
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required; use runtime.device=cpu for a small smoke run"
            )
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if device_type == "cuda" else "gloo",
            init_method="env://",
            # Async NCCL error handling is enabled by launch.py. This bounded
            # timeout is the final guard against a peer dying inside a DDP
            # forward/backward collective and leaving the other ranks waiting
            # for an entire day.
            timeout=PROCESS_GROUP_TIMEOUT,
        )
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def shutdown_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()
