#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PTQ.common.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GPTQ on OPT endpoints using shared C4 and RefinedWeb caches."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the GPTQ YAML configuration"
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="Prepare/verify the shared C4 and RefinedWeb caches without quantizing",
    )
    args = parser.parse_args()
    run(args.config, method="gptq", prepare_data_only=args.prepare_data_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
