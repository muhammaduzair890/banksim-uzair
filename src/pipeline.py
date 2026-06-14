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
    logger.info(
        f"ARGN datasets — M1 (fraud-only): {len(m1_data):,}  "
        f"M2 (fraud+10%NF): {len(m2_data):,}  M3 (full): {len(m3_data):,}"
    )

    # ── 2. Train ARGN models ──────────────────────────────────────────────────
    if not skip_training:
        logger.info("Training M1, M2, M3 in parallel (one GPU each)…")
        ws_m1, ws_m2, ws_m3 = train_all(m1_data, m2_data, m3_data, fold)
        logger.info("Training complete.")
    else:
        ws_m1 = MODELS_DIR / f"fold_{fold}" / "m1"
        ws_m2 = MODELS_DIR / f"fold_{fold}" / "m2"
        ws_m3 = MODELS_DIR / f"fold_{fold}" / "m3"
        logger.info("Skipping training — using saved models.")

    # ── 3. Generate synthetic data ────────────────────────────────────────────
    if not skip_generation:
        logger.info(f"Generating synthetic fraud pools (20k rows each)…")
        pool_m1, pool_m2, pool_m3 = generate_all(ws_m1, ws_m2, ws_m3, fold)
        logger.info(
            f"Generated — M1: {len(pool_m1):,}  M2: {len(pool_m2):,}  M3: {len(pool_m3):,}"
        )
    else:
        synth_dir = SYNTH_DIR / f"fold_{fold}"
        pool_m1 = pd.read_csv(synth_dir / "pool_m1.csv")
        pool_m2 = pd.read_csv(synth_dir / "pool_m2.csv")
        pool_m3 = pd.read_csv(synth_dir / "pool_m3.csv")
        logger.info("Skipping generation — loaded saved synthetic pools.")

    # ── 4. Evaluate quality ───────────────────────────────────────────────────
    logger.info("Evaluating synthetic data quality…")
    results = evaluate_all(
        pools={"m1": pool_m1, "m2": pool_m2, "m3": pool_m3},
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
