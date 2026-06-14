from pathlib import Path

import pandas as pd

from .config import (
    SYNTH_DIR,
    TARGET_COL,
    POOL_PER_MODEL,
    M2_GEN_BATCH,
    M3_REBAL_PROB,
    GPU_M1,
    GPU_M2,
    GPU_M3,
)


def _load_argn(workspace_dir: Path, device: str):
    from mostlyai.engine import TabularARGN

    argn = TabularARGN(workspace_dir=str(workspace_dir), device=device, verbose=0)
    argn._fitted = True
    return argn


def _collect_fraud(argn, n_generate: int, target_count: int) -> pd.DataFrame:
    """Generate rows and collect fraud samples until target_count is reached."""
    pool: list[pd.DataFrame] = []
    collected = 0
    while collected < target_count:
        batch = argn.sample(n=n_generate)
        fraud_rows = batch[batch[TARGET_COL].astype(str) == "1"]
        pool.append(fraud_rows)
        collected += len(fraud_rows)
    return pd.concat(pool, ignore_index=True).head(target_count)


def generate_m1(workspace_dir: Path, device: str) -> pd.DataFrame:
    """Free generation from M1 (trained on fraud only, should always output fraud=1)."""
    argn = _load_argn(workspace_dir, device)
    # M1 is fraud-only so nearly all samples are fraud — one batch is usually enough
    return _collect_fraud(argn, n_generate=POOL_PER_MODEL, target_count=POOL_PER_MODEL)


def generate_m2(workspace_dir: Path, device: str) -> pd.DataFrame:
    """Batched generation from M2 (fraud + 10% NF), filtering fraud rows each batch."""
    argn = _load_argn(workspace_dir, device)
    return _collect_fraud(argn, n_generate=M2_GEN_BATCH, target_count=POOL_PER_MODEL)


def generate_m3(workspace_dir: Path, device: str) -> pd.DataFrame:
    """Rebalanced generation from M3 (full data), forcing 50% fraud at sampling time."""
    from mostlyai.engine.domain import RebalancingConfig

    argn = _load_argn(workspace_dir, device)
    rebal = RebalancingConfig(column=TARGET_COL, probabilities={"1": M3_REBAL_PROB})

    pool: list[pd.DataFrame] = []
    collected = 0
    n_generate = POOL_PER_MODEL * 3
    while collected < POOL_PER_MODEL:
        batch = argn.sample(n=n_generate, rebalancing=rebal)
        fraud_rows = batch[batch[TARGET_COL].astype(str) == "1"]
        pool.append(fraud_rows)
        collected += len(fraud_rows)
        n_generate = POOL_PER_MODEL * 4   # scale up if yield is low

    return pd.concat(pool, ignore_index=True).head(POOL_PER_MODEL)


def generate_all(
    ws_m1: Path, ws_m2: Path, ws_m3: Path, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = SYNTH_DIR / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_m1 = generate_m1(ws_m1, f"cuda:{GPU_M1}")
    pool_m1.to_csv(out_dir / "pool_m1.csv", index=False)

    pool_m2 = generate_m2(ws_m2, f"cuda:{GPU_M2}")
    pool_m2.to_csv(out_dir / "pool_m2.csv", index=False)

    pool_m3 = generate_m3(ws_m3, f"cuda:{GPU_M3}")
    pool_m3.to_csv(out_dir / "pool_m3.csv", index=False)

    return pool_m1, pool_m2, pool_m3
