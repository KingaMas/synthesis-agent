#!/usr/bin/env python
"""
M3: contrastive composition encoder trained on recipe-overlap supervision.

Runs in .venv-rr (torch; imports nothing from src/). Input is the .npz
written by run_m3_validation.py prep; output is an .npz of L2-normalised
embeddings consumed by run_m3_validation.py eval.

Model: MLP over [element-fraction ; standardised Magpie] features with a
soft-target InfoNCE loss — for each anchor, the softmax over in-batch
cosine similarities is pushed toward the row-normalised precursor-set
Jaccard overlaps (the recipe-overlap supervision from the handoff spec).
Batches pair each anchor with one sampled high-overlap partner so
positives are never vanishingly rare.

Early stopping selects on exact validation formula-SRO@5 computed from
the vocab-restricted label sets — the benchmark metric itself, not a
proxy. Val queries with no in-vocab precursors are excluded from the
early-stop metric (they still get embedded).

Usage
-----
    .venv-rr/bin/python scripts/m3_train_contrastive.py \
        --data <arrays.npz> --out <embeddings.npz>
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np


def overlap_matrix(y_a: np.ndarray, y_b: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard between binary label rows (float32)."""
    inter = y_a @ y_b.T
    sizes_a = y_a.sum(axis=1, keepdims=True)
    sizes_b = y_b.sum(axis=1, keepdims=True)
    union = sizes_a + sizes_b.T - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.where(union > 0, inter / union, 0.0)
    return o.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-set", default="val")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-anchors", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--partner-min-overlap", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop patience, in eval steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    blobs = np.load(args.data)
    mean, std = blobs["magpie_mean"], blobs["magpie_std"]

    def features(name: str) -> torch.Tensor:
        magpie = (blobs[f"{name}_magpie"] - mean) / std
        magpie = np.nan_to_num(magpie, nan=0.0)
        return torch.tensor(
            np.hstack([blobs[f"{name}_comp"], magpie]), dtype=torch.float32
        )

    x_train = features("train")
    x_val = features(args.val_set)
    y_train = blobs["train_y"]
    y_val = blobs[f"{args.val_set}_y"]
    n_train = len(x_train)

    print(f"train {n_train}, val {len(x_val)}, feature dim {x_train.shape[1]}")

    print("precomputing train-train and val-train overlap matrices ...")
    o_train = overlap_matrix(y_train, y_train)
    np.fill_diagonal(o_train, 0.0)
    o_val = torch.tensor(overlap_matrix(y_val, y_train))
    val_mask = y_val.sum(axis=1) > 0
    partners = [
        np.flatnonzero(o_train[i] >= args.partner_min_overlap)
        for i in range(n_train)
    ]
    o_train_t = torch.tensor(o_train)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], args.hidden), nn.PReLU(),
        nn.Linear(args.hidden, args.hidden // 2), nn.PReLU(),
        nn.Linear(args.hidden // 2, args.emb_dim),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def embed(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(model(x), p=2, dim=1)

    @torch.no_grad()
    def val_fsro_at_5() -> float:
        model.eval()
        sims = embed(x_val) @ embed(x_train).T
        top5 = sims.topk(5, dim=1).indices
        scores = o_val.gather(1, top5).mean(dim=1).numpy()
        return float(scores[val_mask].mean())

    rng = np.random.default_rng(args.seed)
    steps = max(1, n_train // args.batch_anchors)
    best, best_state, since_best = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(steps):
            anchors = rng.choice(n_train, size=args.batch_anchors,
                                 replace=False)
            batch = set(anchors.tolist())
            for a in anchors:
                if len(partners[a]):
                    batch.add(int(rng.choice(partners[a])))
            idx = torch.tensor(sorted(batch))

            z = embed(x_train[idx])
            sim = z @ z.T / args.tau
            sim.fill_diagonal_(-1e4)
            target = o_train_t[idx][:, idx].clone()
            row_sums = target.sum(dim=1)
            keep = row_sums > 0
            if not keep.any():
                continue
            target = target[keep] / row_sums[keep].unsqueeze(1)
            loss = -(target * F.log_softmax(sim[keep], dim=1)).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)

        if (epoch + 1) % args.eval_every == 0:
            fsro = val_fsro_at_5()
            marker = ""
            if fsro > best:
                best, since_best = fsro, 0
                best_state = copy.deepcopy(model.state_dict())
                marker = " *"
            else:
                since_best += 1
            print(f"epoch {epoch + 1}: loss {total / steps:.4f} "
                  f"val fSRO@5 {fsro:.4f}{marker}", flush=True)
            if since_best >= args.patience:
                print("early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    sets = sorted({k.rsplit("_", 1)[0] for k in blobs.files
                   if k.endswith("_comp")})
    out = {"_best_val_fsro5": np.array([best])}
    with torch.no_grad():
        for name in sets:
            out[f"{name}_emb"] = embed(features(name)).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"embeddings saved to {args.out} (best val fSRO@5 {best:.4f})")


if __name__ == "__main__":
    main()
