from __future__ import annotations

import os
import random
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import wandb

from .config import RANDOM_STATE, resolve_resource_config

warnings.filterwarnings("ignore", category=RuntimeWarning)

IS_POSTPROCESSING_WORKER = os.environ.get("POSTPROCESSING_WORKER", "0") == "1"
_RESOURCES = resolve_resource_config(IS_POSTPROCESSING_WORKER)
N_CPUS = _RESOURCES["n_cpus"]
N_INTEROP = _RESOURCES["n_interop"]
NUM_WORKERS = _RESOURCES["num_workers"]
POSTPROCESSING_WORKERS = _RESOURCES["postprocessing_workers"]

os.environ["OMP_NUM_THREADS"] = str(N_CPUS)
os.environ["MKL_NUM_THREADS"] = str(N_CPUS)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

try:
    torch.set_num_threads(N_CPUS)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(N_INTEROP)
except RuntimeError:
    pass

if IS_POSTPROCESSING_WORKER:
    DEVICE = torch.device("cpu")
    REQUIRE_GPU = False
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") != "0"

if REQUIRE_GPU and DEVICE.type != "cuda":
    raise RuntimeError(
        "No CUDA GPU found. Request a GPU in SLURM or set REQUIRE_GPU=0 "
        "for an intentional CPU test."
    )

USE_AMP = True

# Enforce deterministic PyTorch kernels. CUBLAS_WORKSPACE_CONFIG must already
# have been set by sweep.py before torch is imported.
torch.use_deterministic_algorithms(True)

if DEVICE.type == "cuda" and not IS_POSTPROCESSING_WORKER:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reproducibility_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "wandb": wandb.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(DEVICE),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "amp_enabled": bool(USE_AMP),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "random_state": RANDOM_STATE,
        "os_cpu_count": os.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "n_cpus": N_CPUS,
        "torch_num_interop_threads": N_INTEROP,
        "dataloader_workers": NUM_WORKERS,
        "core_search_workers": POSTPROCESSING_WORKERS,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return metadata
