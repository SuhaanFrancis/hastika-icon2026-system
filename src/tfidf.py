"""Word + character TF-IDF with class-weighted logistic regression.

  python -m src.tfidf dev  --task A --train data/binary_train.csv --out outputs/A_tfidf_dev.npy
  python -m src.tfidf test --task A --train data/binary_train.csv --test data/binary_test.csv \
                           --out outputs/A_tfidf_test.npy
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .common import load_split, split_dev


def fit_predict_proba(train_texts, train_labels, eval_texts, task: str, c: float = 1.0, min_df: int = 1):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df)
    x_train = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)]).tocsr()
    x_eval = hstack([word.transform(eval_texts), char.transform(eval_texts)]).tocsr()
    clf = LogisticRegression(C=c, max_iter=2000 if task == "A" else 3000, class_weight="balanced")
    clf.fit(x_train, train_labels)
    return clf.predict_proba(x_eval)  # columns follow the label ids 0..K-1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("dev", "test"):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=["A", "B"], required=True)
        p.add_argument("--train", required=True, help="training CSV")
        p.add_argument("--out", required=True, help="where to save the probabilities (.npy)")
        p.add_argument("--c", type=float, default=1.0)
        p.add_argument("--min-df", type=int, default=1)
        if name == "test":
            p.add_argument("--test", required=True, help="test CSV (labels not needed)")
    args = ap.parse_args()

    train = load_split(args.train, args.task)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.mode == "dev":
        tr, dev = split_dev(train)
        probs = fit_predict_proba(tr["text"], tr["label_id"], dev["text"], args.task, args.c, args.min_df)
        f1 = f1_score(dev["label_id"], probs.argmax(1), average="macro")
        print(f"dev macro-F1: {f1:.4f}  (n={len(dev)})")
    else:
        test = load_split(args.test, args.task, require_labels=False)
        probs = fit_predict_proba(train["text"], train["label_id"], test["text"], args.task, args.c, args.min_df)
        print(f"test probabilities: {probs.shape}")
    np.save(args.out, probs)


if __name__ == "__main__":
    main()
