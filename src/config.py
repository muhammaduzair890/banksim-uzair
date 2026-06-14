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
POOL_PER_MODEL = 20_000   # fraud rows collected per pool
M2_GEN_BATCH = 50_000     # generation batch size for M2 (few fraud rows per batch)
M3_REBAL_PROB = 0.5       # forced fraud probability during M3 generation
M2_NONFR_FRAC = 0.10      # non-fraud fraction included in M2 training data

# GPU assignment (one GPU per model for parallel training)
GPU_M1 = 0
GPU_M2 = 1
GPU_M3 = 2
