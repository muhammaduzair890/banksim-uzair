import logging
import math
import multiprocessing as mp
import tempfile
import time
from pathlib import Path

import pandas as pd

from .config import (
    SYNTH_DIR,
    TARGET_COL,
    POOL_PER_MODEL,
    M1_GEN_BATCH,
    M2_GEN_BATCH,
    M3_GEN_BATCH,
    GEN_GPUS,
)

logger = logging.getLogger(__name__)


def _gen_shard_worker(
    workspace_dir: str,
    device: str,
    target_count: int,
    n_generate: int,
    out_path: str,
) -> None:
    """
    Worker (one per GPU): load the trained model from its workspace, generate
    until `target_count` fraud rows are collected, write the shard to parquet.

    Module-level so multiprocessing 'spawn' can pickle it.
    """
    from mostlyai.engine import TabularARGN

    argn = TabularARGN(workspace_dir=workspace_dir, device=device, verbose=0)
    argn._fitted = True  # model is already trained on disk; skip the fitted guard

    pool: list[pd.DataFrame] = []
    collected = 0
    while collected < target_count:
        batch = argn.sample(n_samples=n_generate)
        fraud_rows = batch[batch[TARGET_COL].astype(float).astype(int) == 1]
        pool.append(fraud_rows)
        collected += len(fraud_rows)

    pd.concat(pool, ignore_index=True).head(target_count).to_parquet(out_path, index=False)


def _parallel_generate(
    workspace_dir: Path,
    target_count: int,
    n_generate: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Shard fraud generation across all GEN_GPUS. Each GPU independently collects
    ~target_count/len(GEN_GPUS) fraud rows; shards are concatenated at the end.
    """
    gpus = GEN_GPUS
    n = len(gpus)
    per_gpu = math.ceil(target_count / n)

    ctx = mp.get_context("spawn")
    tmpdir = Path(tempfile.mkdtemp(prefix=f"gen_{model_name}_"))
    procs = []
    out_paths = []

    start = time.time()
    for i, gpu in enumerate(gpus):
        out = tmpdir / f"shard_{i}.parquet"
        out_paths.append(out)
        p = ctx.Process(
            target=_gen_shard_worker,
            args=(str(workspace_dir), f"cuda:{gpu}", per_gpu, n_generate, str(out)),
            name=f"gen-{model_name}-gpu{gpu}",
        )
        p.start()
        procs.append(p)

    logger.info(
        f"{model_name}: sharded generation across {n} GPUs "
        f"({per_gpu:,} fraud rows/GPU, batch={n_generate:,})"
    )

    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(p.name)

    if failed:
        raise RuntimeError(f"{model_name} generation failed on: {failed}")

    shards = [pd.read_parquet(o) for o in out_paths]
    result = pd.concat(shards, ignore_index=True).head(target_count)
    logger.info(
        f"{model_name}: collected {len(result):,} fraud rows in "
        f"{(time.time() - start) / 60:.1f} min"
    )
    return result


def generate_m1(workspace_dir: Path) -> pd.DataFrame:
    """M1 (fraud-only): nearly all samples are fraud, so harvesting is fast."""
    return _parallel_generate(workspace_dir, POOL_PER_MODEL, M1_GEN_BATCH, "m1")


def generate_m2(workspace_dir: Path) -> pd.DataFrame:
    """M2 (fraud + 10% NF): ~13% of samples are fraud."""
    return _parallel_generate(workspace_dir, POOL_PER_MODEL, M2_GEN_BATCH, "m2")


def generate_m3(workspace_dir: Path) -> pd.DataFrame:
    """
    M3 (full data) at its natural learned fraud rate — no rebalancing.

    M3 emits the ~1.2-1.5% fraud rate it learned, so reaching 20k fraud rows
    requires generating ~1.4-1.6M rows. This is the slowest stage, which is why
    generation is sharded across all GPUs.
    """
    return _parallel_generate(workspace_dir, POOL_PER_MODEL, M3_GEN_BATCH, "m3")


def generate_all(
    ws_m1: Path, ws_m2: Path, ws_m3: Path, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = SYNTH_DIR / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each model is generated in turn, but each fully shards across all GPUs.
    pool_m1 = generate_m1(ws_m1)
    pool_m1.to_csv(out_dir / "pool_m1.csv", index=False)

    pool_m2 = generate_m2(ws_m2)
    pool_m2.to_csv(out_dir / "pool_m2.csv", index=False)

    pool_m3 = generate_m3(ws_m3)
    pool_m3.to_csv(out_dir / "pool_m3.csv", index=False)

    return pool_m1, pool_m2, pool_m3
