import logging
import multiprocessing as mp
import time
from pathlib import Path

from .config import LOGS_DIR, MODELS_DIR, M1_MAX_EPOCHS, M2_MAX_EPOCHS, M3_MAX_EPOCHS, GPU_M1, GPU_M2, GPU_M3

logger = logging.getLogger(__name__)

MODEL_EPOCHS = {"m1": M1_MAX_EPOCHS, "m2": M2_MAX_EPOCHS, "m3": M3_MAX_EPOCHS}
MODEL_GPUS = {"m1": GPU_M1, "m2": GPU_M2, "m3": GPU_M3}


def _train_worker(df, workspace_dir: str, max_epochs: int, device: str, log_path: str) -> None:
    """Module-level function so multiprocessing spawn can import it."""
    from mostlyai.engine import TabularARGN

    # Redirect this subprocess's output to its own log file so ARGN's
    # epoch-by-epoch progress is captured rather than lost.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )

    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    argn = TabularARGN(
        max_epochs=max_epochs,
        workspace_dir=str(ws),
        device=device,
        verbose=1,
    )
    argn.fit(df)


def train_all(
    datasets: dict, fold: int, models=("m1", "m2", "m3")
) -> dict:
    """Train the selected models in parallel, each on its own GPU.

    `datasets` maps model name ("m1"/"m2"/"m3") to its training DataFrame.
    Returns a dict mapping each trained model name to its workspace Path.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    workspaces = {name: MODELS_DIR / f"fold_{fold}" / name for name in models}

    specs = [
        (f"argn-{name}", datasets[name], workspaces[name], MODEL_EPOCHS[name], MODEL_GPUS[name])
        for name in models
    ]

    ctx = mp.get_context("spawn")
    processes = []
    start_times = {}

    for name, data, ws, epochs, gpu in specs:
        log_path = LOGS_DIR / f"fold_{fold}_{name}.log"
        p = ctx.Process(
            target=_train_worker,
            args=(data, str(ws), epochs, f"cuda:{gpu}", str(log_path)),
            name=name,
        )
        p.start()
        processes.append(p)
        start_times[name] = time.time()
        logger.info(f"Started {name} on cuda:{gpu} (max_epochs={epochs}) — logs: {log_path}")

    failed = []
    for p in processes:
        p.join()
        elapsed = time.time() - start_times[p.name]
        if p.exitcode == 0:
            logger.info(f"{p.name} finished in {elapsed / 60:.1f} min")
        else:
            logger.error(f"{p.name} failed after {elapsed / 60:.1f} min (exit code {p.exitcode})")
            failed.append(p.name)

    if failed:
        raise RuntimeError(f"ARGN training failed for: {failed}")

    return workspaces
