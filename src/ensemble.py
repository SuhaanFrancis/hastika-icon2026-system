"""Weighted probability averaging: search the weights on the dev split, apply them to test.

  python -m src.ensemble search --task A --train data/binary_train.csv \
         --dev-probs outputs/A_tfidf_dev.npy outputs/A_muril_dev.npy --out outputs/A_weights.json
  python -m src.ensemble apply  --task A --test data/binary_test.csv \
         --test-probs outputs/A_tfidf_test.npy outputs/A_muril_test.npy \
         --weights 0.75 0.25 --out outputs/A_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .common import LABELS, load_split, split_dev


def combine(probs_list, weights):
    return sum(w * p for w, p in zip(weights, probs_list))


def macro_f1(probs_list, weights, y):
    return f1_score(y, combine(probs_list, weights).argmax(1), average="macro")


def search_grid(probs_list, y, step=0.02):
    """Two models: sweep the weight of the second model."""
    best = (-1.0, None)
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        weights = [1.0 - w, w]
        score = macro_f1(probs_list, weights, y)
        if score > best[0]:
            best = (score, weights)
    return best


def search_random(probs_list, y, n_trials=3000, seed=42):
    """Three or more models: random weight vectors from a uniform Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    best = (-1.0, None)
    for _ in range(n_trials):
        weights = rng.dirichlet(np.ones(len(probs_list)))
        score = macro_f1(probs_list, weights, y)
        if score > best[0]:
            best = (score, weights.tolist())
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("search")
    s.add_argument("--task", choices=["A", "B"], required=True)
    s.add_argument("--train", required=True, help="training CSV (used to recompute the dev labels)")
    s.add_argument("--dev-probs", nargs="+", required=True, help="dev probabilities, one .npy per model")
    s.add_argument("--n-trials", type=int, default=3000)
    s.add_argument("--step", type=float, default=0.02)
    s.add_argument("--out", required=True)

    a = sub.add_parser("apply")
    a.add_argument("--task", choices=["A", "B"], required=True)
    a.add_argument("--test", required=True, help="test CSV (only the id column is used)")
    a.add_argument("--test-probs", nargs="+", required=True)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", nargs="+", type=float)
    g.add_argument("--weights-json")
    a.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "search":
        _, dev = split_dev(load_split(args.train, args.task))
        y = dev["label_id"].values
        probs = [np.load(p) for p in args.dev_probs]
        for p, name in zip(probs, args.dev_probs):
            assert len(p) == len(y), f"{name}: {len(p)} rows, expected {len(y)} (same split and seed needed)"
            print(f"{name}: single-model dev macro-F1 {f1_score(y, p.argmax(1), average='macro'):.4f}")
        if len(probs) == 2:
            score, weights = search_grid(probs, y, args.step)
            method = f"grid(step={args.step})"
        else:
            score, weights = search_random(probs, y, args.n_trials)
            method = f"random({args.n_trials} Dirichlet draws)"
        print(f"best {method}: weights={np.round(weights, 3).tolist()}  dev macro-F1={score:.4f}")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"weights": weights, "dev_macro_f1": score, "method": method}, f, indent=2)
    else:
        weights = args.weights
        if args.weights_json:
            with open(args.weights_json) as f:
                weights = json.load(f)["weights"]
        probs = [np.load(p) for p in args.test_probs]
        assert len(weights) == len(probs), "need one weight per probability file"
        assert abs(sum(weights) - 1.0) < 1e-3, "weights must sum to 1"
        ids = load_split(args.test, args.task, require_labels=False)["id"]
        pred = combine(probs, weights).argmax(1)
        out = pd.DataFrame({"id": ids, "label": [LABELS[args.task][i] for i in pred]})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8")
        print(f"wrote {args.out}: {len(out)} rows")
        print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
