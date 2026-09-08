#!/usr/bin/env python3
"""Executable entry point for the Bayesian W&B LOAO sweep."""
from __future__ import annotations

import os

# These settings must be established before importing PyTorch. They are also
# applied when multiprocessing ``spawn`` re-imports this entry-point module.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

if os.environ.get("POSTPROCESSING_WORKER", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"

from hp_optimization.experiment import parse_args, run_trial


if __name__ == "__main__":
    run_trial(parse_args())
