import json
import logging
import logging.handlers
from pathlib import Path

import pandas as pd

from .config import LOGS_DIR, MODELS_DIR, SYNTH_DIR, RESULTS_DIR
from .data import load_fold, build_argn_datasets
from .train_argn import train_all
from .generate import generate_all
from .evaluate import evaluate_all


def _setup_logger(fold: int) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"fold_{fold}.log"

    logger = logging.getLogger(f"banksim.fold_{fold}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


def run_fold(
    fold: int,
    skip_training: bool = False,
    skip_generation: bool = False,
    models=("m1", "m2", "m3"),
) -> dict:
    logger = _setup_logger(fold)
    logger.info(f"{'='*40} Fold {fold} {'='*40}")

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    train, test = load_fold(fold)
    fraud_rate = train["fraud"].mean()
    logger.info(
        f"Loaded fold {fold}: train={len(train):,}  test={len(test):,}  "
        f"fraud_rate={fraud_rate:.4%}"
    )

    m1_data, m2_data, m3_data = build_argn_datasets(train)
    all_datasets = {"m1": m1_data, "m2": m2_data, "m3": m3_data}
    datasets = {name: all_datasets[name] for name in models}
    logger.info(
        f"ARGN datasets — M1 (fraud-only): {len(m1_data):,}  "
        f"M2 (fraud+10%NF): {len(m2_data):,}  M3 (full): {len(m3_data):,}  "
        f"| training models: {list(models)}"
    )

    # ── 2. Train ARGN models ──────────────────────────────────────────────────
    if not skip_training:
        logger.info(f"Training {list(models)} in parallel (one GPU each)…")
        workspaces = train_all(datasets, fold, models)
        logger.info("Training complete.")
    else:
        workspaces = {name: MODELS_DIR / f"fold_{fold}" / name for name in models}
        logger.info("Skipping training — using saved models.")

    # ── 3. Generate synthetic data ────────────────────────────────────────────
    if not skip_generation:
        logger.info(f"Generating synthetic fraud pools (20k rows each)…")
        pools = generate_all(workspaces, fold, models)
        logger.info(
            "Generated — " + "  ".join(f"{k}: {len(v):,}" for k, v in pools.items())
        )
    else:
        synth_dir = SYNTH_DIR / f"fold_{fold}"
        pools = {name: pd.read_csv(synth_dir / f"pool_{name}.csv") for name in models}
        logger.info("Skipping generation — loaded saved synthetic pools.")

    # ── 4. Evaluate quality ───────────────────────────────────────────────────
    logger.info("Evaluating synthetic data quality…")
    results = evaluate_all(
        pools=pools,
        train=train,
        test=test,
        fold=fold,
    )

    # Save per-fold summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"fold_{fold}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Fold {fold} summary saved to {summary_path}")

    return results
