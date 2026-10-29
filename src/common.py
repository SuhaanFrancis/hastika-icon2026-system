"""Shared helpers: label sets, data loading and the development split."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import clean_text

LABELS = {
    "A": ["Non-Hate", "Hate"],
    "B": ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"],
}
LABEL_COLUMN = {"A": "Label", "B": "Hate Category"}
SEED = 42
DEV_FRACTION = 0.15


def load_split(path: str, task: str, require_labels: bool = True) -> pd.DataFrame:
    """Read a HASTIKA CSV and return columns id, text (cleaned) and, if requested, label_id."""
    df = pd.read_csv(path, encoding="utf-8-sig")  # the released files start with a BOM
    out = pd.DataFrame({"id": df["id"].values, "text": df["Comment"].map(clean_text).values})
    if require_labels:
        label2id = {name: i for i, name in enumerate(LABELS[task])}
        labels = df[LABEL_COLUMN[task]].map(label2id)
        if labels.isna().any():
            bad = df.loc[labels.isna(), LABEL_COLUMN[task]].unique()[:5]
            raise ValueError(f"Unexpected labels in {path}: {list(bad)}")
        out["label_id"] = labels.astype(int).values
    return out


def split_dev(df: pd.DataFrame, seed: int = SEED, fraction: float = DEV_FRACTION):
    """Stratified train/dev split (85/15, seed 42), as used for all development experiments."""
    train_df, dev_df = train_test_split(df, test_size=fraction, random_state=seed, stratify=df["label_id"])
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)
