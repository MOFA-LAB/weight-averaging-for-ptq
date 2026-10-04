from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PTQ.common.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run activation-aware reference PTQ on OPT endpoints."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="prepare/verify shared C4 and RefinedWeb held-out caches, then exit",
    )
    arguments = parser.parse_args()
    run(arguments.config, method="awq", prepare_data_only=arguments.prepare_data_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
