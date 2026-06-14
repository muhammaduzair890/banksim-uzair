import pandas as pd
from .config import DATA_DIR, TARGET_COL, DROP_COLS, RANDOM_STATE, M2_NONFR_FRAC

# Columns that are categorical codes even though pandas reads them as integers.
# Cast to str so ARGN treats them as discrete categories, not continuous values.
CAT_COLS = ["age"]


def load_fold(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess train/test CSVs for a given fold."""
    train = pd.read_csv(DATA_DIR / f"splits/{fold}/train.csv")
    test = pd.read_csv(DATA_DIR / f"splits/{fold}/test.csv")

    for df in (train, test):
        for col in CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)

    train = train.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)
    test = test.drop(columns=DROP_COLS, errors="ignore").reset_index(drop=True)

    train[TARGET_COL] = train[TARGET_COL].astype(int)
    test[TARGET_COL] = test[TARGET_COL].astype(int)

    return train, test


def build_argn_datasets(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build three training datasets from the full train split:
      M1 – fraud rows only
      M2 – all fraud + 10% non-fraud (stratified by category)
      M3 – full training data
    """
    fraud = train[train[TARGET_COL] == 1].reset_index(drop=True)
    nonfraud = train[train[TARGET_COL] == 0]

    nf_sample = (
        nonfraud.groupby("category", group_keys=False)
        .apply(lambda x: x.sample(frac=M2_NONFR_FRAC, random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )

    m1 = fraud.copy()
    m2 = pd.concat([fraud, nf_sample], ignore_index=True)
    m3 = train.copy()

    return m1, m2, m3
