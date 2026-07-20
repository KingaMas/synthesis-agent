#!/usr/bin/env python
"""
NRE stage 2: formation energies and delta-G score matrix (T3).

Runs in .venv-rr; imports GraphNetwork_prop unmodified from the
Retrieval-Retro clone and loads their pretrained formation-energy
checkpoint (transfer-learned on experimental formation energies).
Graph construction replicates theirs exactly (verified against their
tensors): nodes = recovered 200-dim element rows, fully connected with
self-loops, edge_attr = concat of endpoint features, fc_weight =
composition fractions.

Score follows their calculate_gibbs.py: score(query, corpus entry j) =
Ef(query) - sum(Ef(precursors of j)), ranked ascending. Entries with no
in-vocab precursors or unmappable elements score +inf; unmappable
queries are omitted from the score dict (the eval side reports them).

Usage
-----
    .venv-rr/bin/python scripts/rr_nre_compute.py \
        --nre-arrays <nre_arrays.npz> --mpc-arrays <mpc_arrays.npz> \
        --rr-repo external/Retrieval-Retro \
        --checkpoint "external/rr_dataset/TL_pretrain(formation_exp)_embedder(graphnetwork)_lr(0.0005)_batch_size(256)_hidden(256)_seed(0)_.pt" \
        --element-table external/rr_dataset/element_table.npz \
        --out <nre_scores.npz>
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nre-arrays", type=Path, required=True)
    ap.add_argument("--mpc-arrays", type=Path, required=True)
    ap.add_argument("--rr-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--element-table", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from torch_geometric.data import Batch, Data

    sys.path.insert(0, str(args.rr_repo))
    from models import GraphNetwork_prop

    table = np.load(args.element_table)
    rows = torch.tensor(table["rows"], dtype=torch.float32)
    id_to_row = {int(i): r for i, r in zip(table["ids"], rows)}

    device = torch.device("cpu")
    n_feat = rows.shape[1]
    model = GraphNetwork_prop(args.layers, n_feat, 2 * n_feat,
                              args.hidden, device).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    print("checkpoint loaded")

    blobs = np.load(args.nre_arrays)

    def make_graph(el_ids, fracs, n) -> Data | None:
        ids = el_ids[:n]
        if any(int(i) not in id_to_row for i in ids):
            return None
        x = torch.stack([id_to_row[int(i)] for i in ids])
        pairs = torch.cartesian_prod(torch.arange(n), torch.arange(n))
        edge_index = pairs.T.contiguous()
        edge_attr = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    fc_weight=torch.tensor(fracs[:n], dtype=torch.float32))

    @torch.no_grad()
    def formation_energies(prefix: str) -> np.ndarray:
        el_ids = blobs[f"{prefix}_el_ids"]
        fracs = blobs[f"{prefix}_fracs"]
        n_els = blobs[f"{prefix}_n_els"]
        out = np.full(len(el_ids), np.nan, dtype=np.float32)
        graphs, keep = [], []
        for i in range(len(el_ids)):
            if n_els[i] == 0:
                continue
            g = make_graph(el_ids[i], fracs[i], int(n_els[i]))
            if g is not None:
                graphs.append(g)
                keep.append(i)
        for s in range(0, len(graphs), args.batch_size):
            batch = Batch.from_data_list(graphs[s:s + args.batch_size])
            y, _ = model(batch)
            out[np.array(keep[s:s + args.batch_size])] = y.flatten().numpy()
        print(f"  {prefix}: {len(keep)}/{len(el_ids)} graphs scored")
        return out

    sets = sorted({k.rsplit("_", 2)[0] for k in blobs.files
                   if k.endswith("_el_ids")})
    print(f"sets: {sets}")
    ef = {name: formation_energies(name) for name in sets}

    train_y = np.load(args.mpc_arrays)["train_y"]
    prec_ef = ef["precursors"]
    # sum of precursor formation energies per corpus entry; entries with
    # missing precursor energies or no labels are unusable
    with np.errstate(invalid="ignore"):
        prec_sum = train_y @ np.nan_to_num(prec_ef, nan=0.0)
    usable = (train_y.sum(axis=1) > 0) & ~(train_y @ np.isnan(prec_ef)
                                           ).astype(bool)
    print(f"corpus entries usable for delta-G: {usable.sum()}/{len(usable)}")

    out = {"corpus_usable": usable, "precursors_ef": prec_ef}
    for name in sets:
        if name == "precursors":
            continue
        scores = ef[name][:, None] - prec_sum[None, :]
        scores[:, ~usable] = np.inf
        scores[np.isnan(ef[name])] = np.inf  # whole row for unscored queries
        out[f"{name}_scores"] = scores.astype(np.float32)
        out[f"{name}_query_scored"] = ~np.isnan(ef[name])
        out[f"{name}_ef"] = ef[name]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"scores saved to {args.out}")


if __name__ == "__main__":
    main()
