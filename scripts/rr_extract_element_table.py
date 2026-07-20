#!/usr/bin/env python
"""
Recover Retrieval-Retro's element vocabulary and node features (T3/NRE).

Their dataset ships composition graphs whose construction code is not
public: nodes carry 200-dim element features, comp_fea is an 83-dim
fraction vector over an undocumented element ordering. This script
recovers both from the data itself:

1. id -> symbol, by constraint intersection: every comp_fea id of a
   graph must be an element of one of its labelled precursors
   (year_template.json gives the precursor formulas). 78/82 ids resolve
   to singletons; the order is Pauling electronegativity. The rest are
   fixed by elimination along that order: id24=Am, id25=Cm (O impossible
   between Pu and Hf), id64=H (X=2.20 between P and Ir; H fails the
   precursor constraint because it arrives via water/atmosphere),
   id81=O (X=3.44 between Cl and F; confirmed as the most frequent id,
   82% of graphs). id51 never occurs and stays unmapped.

2. id -> 200-dim feature row, harvested from graphs where a node's
   fc_weight matches exactly one comp_fea fraction. All 82 ids resolve
   with zero cross-graph inconsistencies.

Run in .venv-rr. Output: external/rr_dataset/element_table.npz
(rows, ids) + element_table.json (symbol -> id), consumed by
run_nre_baseline.py / rr_nre_compute.py.
"""

import json
import re
from pathlib import Path

import numpy as np

MANUAL = {24: "Am", 25: "Cm", 64: "H", 81: "O"}
DATASET = Path("external/rr_dataset")


def main():
    import torch

    data = torch.load(DATASET / "year/year_train_mpc.pt",
                      map_location="cpu", weights_only=False)
    template = json.load(open(DATASET / "year/year_template.json"))
    prec_els = [set(re.findall(r"[A-Z][a-z]?", f)) for f in template]

    cand: dict[int, set] = {}
    emb: dict[int, np.ndarray] = {}
    for g in data:
        ids = g.comp_fea.nonzero().flatten().tolist()
        labels = (g.y_lb_all.nonzero().flatten().tolist()
                  or g.y_lb_one.nonzero().flatten().tolist())
        els = set().union(*[prec_els[i] for i in labels]) if labels else set()
        if els:
            for i in ids:
                cand[i] = cand[i] & els if i in cand else els.copy()

        fracs = {i: float(g.comp_fea[i]) for i in ids}
        for n in range(g.x.shape[0]):
            w = float(g.fc_weight[n])
            matches = [i for i in ids if abs(fracs[i] - w) < 1e-6]
            if len(matches) != 1:
                continue
            i, row = matches[0], g.x[n].numpy()
            if i in emb:
                assert np.allclose(emb[i], row, atol=1e-5), f"id {i} inconsistent"
            else:
                emb[i] = row

    mapping = {i: next(iter(s)) for i, s in cand.items() if len(s) == 1}
    mapping.update(MANUAL)
    assert len(mapping) == len(emb) == 82, (len(mapping), len(emb))
    assert set(mapping) == set(emb)

    ids = np.array(sorted(emb))
    np.savez(DATASET / "element_table.npz",
             ids=ids, rows=np.stack([emb[i] for i in ids]))
    with open(DATASET / "element_table.json", "w") as f:
        json.dump({mapping[i]: int(i) for i in sorted(mapping)}, f, indent=1)
    print(f"wrote element table for {len(ids)} elements to {DATASET}")


if __name__ == "__main__":
    main()
