import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SYNTH_DIR = BASE_DIR / "synthetic"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Data
TARGET_COL = "fraud"
DROP_COLS = ["customer", "zipcodeOri", "zipMerchant"]
N_FOLDS = 5
RANDOM_STATE = 42

# ARGN training
M1_MAX_EPOCHS = 50    # fraud-only; small dataset, early stop saves time
M2_MAX_EPOCHS = 100
M3_MAX_EPOCHS = 100

# Generation
POOL_PER_MODEL = 65_000   # fraud rows collected per pool (>= M_max(10) * max fold fraud 6160)
M1_GEN_BATCH = 10_000     # per-shard batch for M1 (~100% fraud)
M2_GEN_BATCH = 25_000     # per-shard batch for M2 (~13% fraud per batch)
M3_GEN_BATCH = 100_000    # per-shard batch for M3 (natural ~1.3% fraud rate)
M2_NONFR_FRAC = 0.10      # non-fraud fraction included in M2 training data

# GPU assignment for training: one model per GPU, trained in parallel.
# Overridable via env (e.g. GPU_M1=2) to pin a run to specific devices.
GPU_M1 = int(os.environ.get("GPU_M1", 0))
GPU_M2 = int(os.environ.get("GPU_M2", 1))
GPU_M3 = int(os.environ.get("GPU_M3", 2))

# GPUs used to shard generation. Generation is embarrassingly parallel, so every
# model's fraud harvest is split across all of these GPUs at once.
# Override via env with a comma-separated list, e.g. GEN_GPUS="2,3".
GEN_GPUS = [int(g) for g in os.environ.get("GEN_GPUS", "0,1,2,3").split(",")]
