"""Fine-tune a transformer encoder and write class probabilities for the dev and test sets.

  python -m src.finetune train --task A --model google/muril-base-cased --train data/binary_train.csv \
         --out-dir outputs/A_muril --epochs 4 --lr 2e-5 --warmup-ratio 0.1
  python -m src.finetune predict --task A --model google/muril-base-cased \
         --checkpoint outputs/A_muril/best.pt --test data/binary_test.csv --out outputs/A_muril_test.npy

`train` keeps the epoch with the best dev macro-F1 (best.pt) and also saves the last epoch (last.pt).
Use --init-checkpoint to continue from a saved checkpoint (see the README for the Task A recipe).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .common import LABELS, load_split, split_dev


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, texts, max_length: int):
    enc = tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict_probs(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in loader:
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks)


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_labels = len(LABELS[args.task])

    train_df, dev_df = split_dev(load_split(args.train, args.task))
    if args.max_train_samples:  # quick smoke test only
        train_df = train_df.sample(args.max_train_samples, random_state=args.seed).reset_index(drop=True)
        dev_df = dev_df.head(max(args.max_train_samples // 4, 16))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tr_ids, tr_mask = encode(tokenizer, train_df["text"], args.max_length)
    dv_ids, dv_mask = encode(tokenizer, dev_df["text"], args.max_length)
    train_loader = DataLoader(
        TensorDataset(tr_ids, tr_mask, torch.tensor(train_df["label_id"].values)),
        batch_size=args.batch_size, shuffle=True,
    )
    dev_loader = DataLoader(
        TensorDataset(dv_ids, dv_mask, torch.tensor(dev_df["label_id"].values)),
        batch_size=args.batch_size, shuffle=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=n_labels)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu"))
    model.to(device)

    weight = None
    if args.class_weights:
        y = train_df["label_id"].values
        if len(np.unique(y)) == n_labels:
            weight = torch.tensor(
                compute_class_weight("balanced", classes=np.arange(n_labels), y=y), dtype=torch.float
            ).to(device)
            print("class weights:", np.round(weight.cpu().numpy(), 3).tolist())
        else:
            print("warning: not all classes present in the training subset; using unweighted loss")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = None
    if args.warmup_ratio > 0:
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    os.makedirs(args.out_dir, exist_ok=True)
    history, best_f1, best_state, best_probs = [], -1.0, None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()
        for step, (x, m, y) in enumerate(train_loader, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(input_ids=x, attention_mask=m).logits, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  batch {step}/{len(train_loader)}", flush=True)
        probs = predict_probs(model, dev_loader, device)
        f1 = float(f1_score(dev_df["label_id"], probs.argmax(1), average="macro"))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader), "dev_macro_f1": f1})
        print(f"epoch {epoch}/{args.epochs} ({time.time() - t0:.0f}s) train loss {total_loss / len(train_loader):.4f} "
              f"dev macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_probs = f1, probs
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("  -> new best", flush=True)

    torch.save(best_state, os.path.join(args.out_dir, "best.pt"))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(args.out_dir, "last.pt"))
    np.save(os.path.join(args.out_dir, "dev_probs_best.npy"), best_probs)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"best dev macro-F1 {best_f1:.4f}; saved to {args.out_dir}")


def predict(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = load_split(args.test, args.task, require_labels=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids, mask = encode(tokenizer, test["text"], args.max_length)
    loader = DataLoader(TensorDataset(ids, mask), batch_size=args.batch_size, shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(LABELS[args.task]))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    probs = predict_probs(model, loader, device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, probs)
    print(f"test probabilities: {probs.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train")
    t.add_argument("--task", choices=["A", "B"], required=True)
    t.add_argument("--model", required=True, help="Hugging Face model name or local path")
    t.add_argument("--train", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--max-length", type=int, default=128)
    t.add_argument("--warmup-ratio", type=float, default=0.0, help="linear warmup + decay if > 0")
    t.add_argument("--class-weights", action="store_true", help="inverse-frequency weighted cross-entropy")
    t.add_argument("--init-checkpoint", help="state dict (.pt) to continue from")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--max-train-samples", type=int, help="use a small subset (smoke test only)")

    p = sub.add_parser("predict")
    p.add_argument("--task", choices=["A", "B"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)

    args = ap.parse_args()
    train(args) if args.mode == "train" else predict(args)


if __name__ == "__main__":
    main()
