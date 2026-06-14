import json
import logging
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR, TARGET_COL

logger = logging.getLogger(__name__)


def _metrics_to_dict(metrics) -> dict:
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    if hasattr(metrics, "__dict__"):
        return metrics.__dict__
    return dict(metrics)


def report_quality(
    syn_data: pd.DataFrame,
    trn_data: pd.DataFrame,
    hol_data: pd.DataFrame,
    model_name: str,
    fold: int,
) -> dict:
    """
    Generate a mostlyai-qa quality report comparing synthetic data against
    train (trn) and holdout/test (hol) distributions.

    All three DataFrames should share the same schema (fraud rows only for
    M1/M2/M3 pools; same columns as the training data).
    """
    from mostlyai import qa

    out_dir = RESULTS_DIR / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / f"{model_name}_quality_report.html"

    _, metrics = qa.report(
        syn_tgt_data=syn_data,
        trn_tgt_data=trn_data,
        hol_tgt_data=hol_data,
        report_path=str(report_path),
    )

    metrics_dict = _metrics_to_dict(metrics)

    with open(out_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)

    logger.info(
        f"Fold {fold} | {model_name} | report saved to {report_path}"
    )
    return metrics_dict


def evaluate_all(
    pools: dict[str, pd.DataFrame],
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
) -> dict[str, dict]:
    """
    Run quality evaluation for each synthetic pool.

    All three pools contain fraud-only rows, so we compare them against
    the fraud subset of train and test for a like-for-like distribution check.
    """
    fraud_train = train[train[TARGET_COL] == 1].reset_index(drop=True)
    fraud_test = test[test[TARGET_COL] == 1].reset_index(drop=True)

    if len(fraud_test) == 0:
        logger.warning(f"Fold {fold}: no fraud rows in test set; using full test for holdout.")
        fraud_test = test

    results = {}
    for name, pool in pools.items():
        logger.info(f"Fold {fold} | evaluating {name} (pool size={len(pool)})")
        results[name] = report_quality(pool, fraud_train, fraud_test, name, fold)

    return results
