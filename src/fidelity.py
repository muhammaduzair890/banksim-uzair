"""
PaySim-style fidelity analysis for BankSim, run WITHOUT retraining.

Compares the already-generated synthetic fraud pools (synthetic/fold_*/pool_m*.csv)
against the real fraud rows in each fold's training split, on the marginal
"fraud-signature" properties that matter for BankSim.

BankSim has no transaction type / balance / isFlaggedFraud columns, so the
PaySim properties (acct-drained, zero-drain, mule accounts) are replaced by the
BankSim equivalents: category concentration, legit-category leakage, amount
stats, gender skew, age skew.
"""
import argparse
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, SYNTH_DIR, RESULTS_DIR, N_FOLDS, TARGET_COL

MODELS = ["m1", "m2", "m3"]

# Categories that dominate legitimate traffic and almost never carry fraud.
# Leakage into these is the BankSim analog of "non-fraud types leak in".
LEGIT_CATEGORIES = ["es_transportation", "es_food"]

# Categories that carry the bulk of real fraud.
FRAUD_CORE = ["es_sportsandtoys", "es_health"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["age", "gender", "category", "merchant"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip("'")
    return df


def _props(df: pd.DataFrame) -> dict:
    n = len(df)
    cat = df["category"].value_counts(normalize=True)
    return {
        "n": n,
        "amount_mean": df["amount"].mean(),
        "amount_median": df["amount"].median(),
        "amount_max": df["amount"].max(),
        "fraud_core_share": cat.reindex(FRAUD_CORE).fillna(0).sum(),
        "legit_leak_share": cat.reindex(LEGIT_CATEGORIES).fillna(0).sum(),
        "n_categories": df["category"].nunique(),
        "gender_F_share": (df["gender"] == "F").mean(),
        "age_top_share": df["age"].value_counts(normalize=True).max(),
        "n_merchants": df["merchant"].nunique(),
        "merchant_top_share": df["merchant"].value_counts(normalize=True).max(),
    }


def analyze_fold(fold: int) -> dict:
    train = _clean(pd.read_csv(DATA_DIR / "splits" / str(fold) / "train.csv"))
    real = train[train[TARGET_COL] == 1].reset_index(drop=True)
    out = {"real": _props(real)}
    for m in MODELS:
        path = SYNTH_DIR / f"fold_{fold}" / f"pool_{m}.csv"
        if not path.exists():
            continue
        syn = _clean(pd.read_csv(path))
        syn = syn[syn[TARGET_COL] == 1].reset_index(drop=True)
        out[m] = _props(syn)
    return out


def aggregate(per_fold: list[dict]) -> dict:
    keys = per_fold[0]["real"].keys()
    agg = {}
    for src in ["real"] + MODELS:
        rows = [f[src] for f in per_fold if src in f]
        if not rows:
            continue
        agg[src] = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    return agg


def _fmt(v: float, k: str) -> str:
    if k == "n" or k.startswith("n_"):
        return f"{v:,.0f}"
    if k.endswith("_share"):
        return f"{v*100:.1f}%"
    return f"{v:,.0f}" if v >= 1000 else f"{v:,.1f}"


PROP_LABELS = [
    ("amount_mean", "amount mean"),
    ("amount_median", "amount median"),
    ("amount_max", "amount max"),
    ("fraud_core_share", "fraud-core categories (sportsandtoys+health)"),
    ("legit_leak_share", "legit-category leak (transportation+food)"),
    ("n_categories", "# distinct categories"),
    ("gender_F_share", "gender == F"),
    ("age_top_share", "top age-bucket share"),
    ("n_merchants", "# distinct merchants"),
    ("merchant_top_share", "top merchant share"),
]


def print_table(agg: dict) -> None:
    srcs = [s for s in ["real", "m1", "m2", "m3"] if s in agg]
    header = ["Property", "Real fraud"] + [s.upper() for s in srcs if s != "real"]
    widths = [44, 14] + [12] * (len(header) - 2)

    def row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(row(header))
    print(row(["-" * w for w in widths]))
    for k, label in PROP_LABELS:
        cells = [label, _fmt(agg["real"][k], k)]
        for s in srcs:
            if s == "real":
                continue
            cells.append(_fmt(agg[s][k], k))
        print(row(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None,
                    help="Single fold to analyze; default = all folds averaged.")
    args = ap.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    per_fold = []
    for fold in folds:
        if not (SYNTH_DIR / f"fold_{fold}").exists():
            continue
        per_fold.append(analyze_fold(fold))

    if not per_fold:
        print("No synthetic pools found.")
        return

    agg = aggregate(per_fold)
    scope = f"fold {args.fold}" if args.fold is not None else f"{len(per_fold)} folds (averaged)"
    print(f"\nBankSim fraud fidelity — {scope}\n")
    print_table(agg)

    out = RESULTS_DIR / "fidelity.json"
    import json
    out.write_text(json.dumps(agg, indent=2, default=float))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
