import multiprocessing as mp
from pathlib import Path

from .config import MODELS_DIR, M1_MAX_EPOCHS, M2_MAX_EPOCHS, M3_MAX_EPOCHS, GPU_M1, GPU_M2, GPU_M3


def _train_worker(df, workspace_dir: str, max_epochs: int, device: str) -> None:
    """Module-level function so multiprocessing spawn can import it."""
    from mostlyai.engine import TabularARGN

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
    m1_data, m2_data, m3_data, fold: int
) -> tuple[Path, Path, Path]:
    """Train M1, M2, M3 in parallel, each on its own GPU."""
    ws_m1 = MODELS_DIR / f"fold_{fold}" / "m1"
    ws_m2 = MODELS_DIR / f"fold_{fold}" / "m2"
    ws_m3 = MODELS_DIR / f"fold_{fold}" / "m3"

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_train_worker,
            args=(m1_data, str(ws_m1), M1_MAX_EPOCHS, f"cuda:{GPU_M1}"),
            name="argn-m1",
        ),
        ctx.Process(
            target=_train_worker,
            args=(m2_data, str(ws_m2), M2_MAX_EPOCHS, f"cuda:{GPU_M2}"),
            name="argn-m2",
        ),
        ctx.Process(
            target=_train_worker,
            args=(m3_data, str(ws_m3), M3_MAX_EPOCHS, f"cuda:{GPU_M3}"),
            name="argn-m3",
        ),
    ]

    for p in processes:
        p.start()

    failed = []
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append(p.name)

    if failed:
        raise RuntimeError(f"ARGN training failed for: {failed}")

    return ws_m1, ws_m2, ws_m3
