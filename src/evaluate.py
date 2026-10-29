"""Score a predictions file against a CSV that contains the gold labels.

  python -m src.evaluate --task B --pred outputs/B_predictions.csv --gold data/multiclass_test_gold.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import LABEL_COLUMN, LABELS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--pred", required=True, help="CSV with columns id,label")
    ap.add_argument("--gold", required=True, help="CSV with the gold labels")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred, encoding="utf-8-sig")
    gold = pd.read_csv(args.gold, encoding="utf-8-sig")[["id", LABEL_COLUMN[args.task]]]
    m = gold.merge(pred, on="id", how="left")
    if m["label"].isna().any():
        raise SystemExit(f"{int(m['label'].isna().sum())} gold ids have no prediction")
    y, p = m[LABEL_COLUMN[args.task]], m["label"]
    names = LABELS[args.task]
    print(classification_report(y, p, labels=names, digits=3))
    print(pd.DataFrame(confusion_matrix(y, p, labels=names), index=names, columns=names))
    print(f"\nmacro-F1 {f1_score(y, p, labels=names, average='macro'):.4f}   accuracy {accuracy_score(y, p):.4f}   n={len(m)}")


if __name__ == "__main__":
    main()
