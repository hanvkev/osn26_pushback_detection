#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import wandb
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=SCRIPT_DIR / "sweep_30runs.yaml",
    )
    parser.add_argument("--project", default="pushback-detection")
    parser.add_argument("--entity", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project,
        entity=args.entity,
    )
    prefix = f"{args.entity}/{args.project}/" if args.entity else f"{args.project}/"
    print(f"\nSWEEP_ID={prefix}{sweep_id}")
    print("Start the W&B agent with:")
    print(f"  ./run_agent.sh {prefix}{sweep_id} 30")


if __name__ == "__main__":
    main()
