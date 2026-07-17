#!/usr/bin/env python
"""
Train Retrieval-Retro's MPC retriever on our corpus (T3).

Runs in the torch environment (.venv-rr: torch + torch_geometric), NOT
the project venv — it deliberately imports nothing from src/. Input is
the .npz written by src/evaluation/mpc_data.py; output is an .npz of
MPC composition embeddings per set, consumed back in the project venv
by scripts/run_mpc_baseline.py.

The model, losses and training protocol are imported unmodified from
the cloned Retrieval-Retro repo (NeurIPS 2024, github.com/HeewoongNoh/
Retrieval-Retro): MPC with hidden=64/emb=32, AdamW lr 5e-4 wd 5e-4,
batch 32, adaptive multi-task loss (CircleLoss + reconstruction MSE).
Deviations, forced by CPU-only hardware: torch.Tensor.cuda is patched
to a no-op, default epoch budget is 300 (not 1000) with early stopping
on validation micro-recall@10 every 5 epochs, patience 6 evals.

Usage
-----
    .venv-rr/bin/python scripts/rr_train_mpc.py \
        --data <arrays.npz> --rr-repo <path> --out <embeddings.npz>
"""

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--rr-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-set", default="val",
                    help="query-set name used for early stopping")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=6,
                    help="early-stop patience, in eval steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self  # their code hardcodes .cuda()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.path.insert(0, str(args.rr_repo))
    from models import MPC, CircleLoss, MultiLossLayer

    blobs = np.load(args.data)
    train_comp = blobs["train_comp"]
    train_y = blobs["train_y"]
    sets = sorted({k.rsplit("_", 1)[0] for k in blobs.files})
    device = torch.device("cpu")

    # Drop train rows with no in-vocab precursor labels (no completion
    # signal, destabilises CircleLoss) and rows reserved for early
    # stopping (train_exclude), which stay in the embedded corpus.
    keep = train_y.sum(axis=1) > 0
    if "train_exclude" in blobs.files:
        keep &= ~blobs["train_exclude"]
    print(f"train rows {len(train_y)}, trained on {int(keep.sum())}")

    def to_data(comp_row, y_row):
        y = torch.tensor(y_row).unsqueeze(0)
        return Data(
            x=torch.zeros(1, 1),
            comp_fea=torch.tensor(comp_row),
            y_lb_one=y,
            y_multiple=y,
            y_multiple_len=torch.tensor([1]),
        )

    train_data = [to_data(c, y) for c, y in
                  zip(train_comp[keep], train_y[keep])]
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    input_dim, output_dim = train_comp.shape[1], train_y.shape[1]
    model = MPC(input_dim, args.hidden, output_dim, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()
    circle_loss = CircleLoss()
    adaptive_loss = MultiLossLayer(["multi-label", "reconstruction"], device)

    val_comp = torch.tensor(blobs[f"{args.val_set}_comp"])
    val_y = blobs[f"{args.val_set}_y"]
    val_mask = val_y.sum(axis=1) > 0

    @torch.no_grad()
    def val_recall_at_10() -> float:
        model.eval()
        # scoring against the full precursor matrix = their kb=None branch
        probs = []
        for i in range(0, len(val_comp), 256):
            batch = [to_data(val_comp[j].numpy(), val_y[j])
                     for j in range(i, min(i + 256, len(val_comp)))]
            from torch_geometric.data import Batch
            p, _, _ = model(Batch.from_data_list(batch))
            probs.append(p)
        probs = torch.cat(probs).numpy()
        recalls = []
        for p_row, y_row in zip(probs[val_mask], val_y[val_mask]):
            top = np.argsort(-p_row)[:10]
            true = np.flatnonzero(y_row)
            recalls.append(len(set(top) & set(true)) / len(true))
        return float(np.mean(recalls))

    best, best_state, since_best = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            multi_label, _, reconstruction = model(batch, None)
            y = batch.y_lb_one.reshape(len(batch.ptr) - 1, -1)
            y_recon = batch.comp_fea.reshape(len(batch.ptr) - 1, -1)
            losses = torch.cat([
                circle_loss(y, multi_label).unsqueeze(0),
                mse_loss(reconstruction, y_recon).unsqueeze(0),
            ], dim=-1)
            loss = adaptive_loss(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)

        if (epoch + 1) % args.eval_every == 0:
            r10 = val_recall_at_10()
            marker = ""
            if r10 > best:
                best, since_best = r10, 0
                best_state = copy.deepcopy(model.state_dict())
                marker = " *"
            else:
                since_best += 1
            print(f"epoch {epoch + 1}: loss {total / len(loader):.4f} "
                  f"val recall@10 {r10:.4f}{marker}", flush=True)
            if since_best >= args.patience:
                print("early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    out = {"_best_val_recall10": np.array([best])}
    with torch.no_grad():
        for name in sets:
            comp = torch.tensor(blobs[f"{name}_comp"])
            out[f"{name}_emb"] = model.comp_encoder(comp).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"embeddings saved to {args.out} (best val recall@10 {best:.4f})")


if __name__ == "__main__":
    main()
