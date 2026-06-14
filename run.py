"""
BankSim ARGN — entry point.

Usage:
    python run.py                          # all 5 folds
    python run.py --fold 0                 # single fold
    python run.py --fold 0 --skip-training # skip training, use saved models
    python run.py --fold 0 --skip-generation --skip-training  # quality eval only
"""
import argparse
import json
import logging

from src.config import N_FOLDS, RESULTS_DIR
from src.pipeline import run_fold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="BankSim ARGN synthetic fraud generation")
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold (0-4). Omit for all folds.")
    parser.add_argument("--skip-training", action="store_true", help="Skip ARGN training, load saved models.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, load saved pools.")
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))

    all_results: dict = {}
    for fold in folds:
        logger.info(f"Starting fold {fold}")
        results = run_fold(
            fold,
            skip_training=args.skip_training,
            skip_generation=args.skip_generation,
        )
        all_results[f"fold_{fold}"] = results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All results saved to {out_path}")


if __name__ == "__main__":
    main()
