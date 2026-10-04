"""Launch one configuration-defined unified pretraining experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config import ConfigError, load_config
from src.data_preparation import DataPreparationError, prepare_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the shared token cache or launch train.py with the number "
            "of local DDP workers declared by the experiment YAML."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Experiment YAML passed unchanged to train.py.",
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help=(
            "Prepare or check the configured token cache, then exit without "
            "initializing CUDA/DDP or creating a training run."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.is_file():
        raise SystemExit(f"configuration does not exist: {config_path}")
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc

    if args.prepare_data_only:
        try:
            prepare_data(config)
        except DataPreparationError as exc:
            raise SystemExit(f"data preparation failed: {exc}") from exc
        return 0

    train_path = project_root / "train.py"
    if not train_path.is_file():
        raise SystemExit(
            f"training entry point is missing: {train_path}; finish the project payload "
            "before launching"
        )

    if config["data"]["preparation"]["prepare_on_launch"]:
        print(
            "[launch] preparing/checking the configured token cache before DDP",
            flush=True,
        )
        try:
            prepare_data(config)
        except DataPreparationError as exc:
            raise SystemExit(f"data preparation failed: {exc}") from exc

    gpu_ids = config["runtime"]["gpu_ids"]
    worker_count = int(config["training"]["expected_world_size"])
    physical_gpus = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = physical_gpus
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={worker_count}",
        str(train_path),
        "--config",
        str(config_path),
    ]
    print(
        f"Launching {worker_count}-process DDP: physical GPUs [{physical_gpus}] "
        f"-> local {[f'cuda:{index}' for index in range(worker_count)]}",
        flush=True,
    )
    print("Command: " + " ".join(command), flush=True)
    return int(subprocess.call(command, cwd=project_root, env=environment))


if __name__ == "__main__":
    raise SystemExit(main())
