#!/usr/bin/env python3
"""Launch training from command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D U-Net spectrum transformer")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config YAML file")
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="Run 2 batches x 1 epoch for sanity check"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg, fast_dev_run=args.fast_dev_run)


if __name__ == "__main__":
    main()
