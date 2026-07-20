#!/usr/bin/env python
"""
Retrieval-Retro MPC baseline on our frozen splits (T3).

Three stages, because MPC needs torch while our harness needs pymatgen
and the two live in different venvs:

  prep  (this venv)     export composition/label arrays for a protocol
  train (.venv-rr)      scripts/rr_train_mpc.py — not this script
  eval  (this venv)     score the saved embeddings with formula-SRO@5
                        against element Jaccard, Wilcoxon+Holm

Protocols
---------
  internal  corpus = train-internal fit (<2014), queries = 2014-15.
            Comparable with the M-ladder validation runs (M1, M2).
  frozen    corpus = full temporal train (<2016), queries = the frozen
            2016+ test side. "es" is a seeded 10% slice of train used
            only for early stopping in the train stage; test queries
            never influence training or model selection.

Usage
-----
    PYTHONPATH=. python scripts/run_mpc_baseline.py prep --protocol internal
    .venv-rr/bin/python scripts/rr_train_mpc.py --data ... --out ...
    PYTHONPATH=. python scripts/run_mpc_baseline.py eval --protocol internal
"""

import argparse
import json
import random
from pathlib import Path


def load_protocol(protocol: str):
    """Returns (corpus, {set_name: query_cases}, eval_set_name)."""
    from src.evaluation.test_set_builder import (
        load_ladder_validation_split,
        load_temporal_split,
    )

    if protocol == "frozen":
        train, test = load_temporal_split()
        rng = random.Random(42)
        es = rng.sample(train, max(1, len(train) // 10))
        return train, {"es": es, "test": test}, "test"

    fit, val = load_ladder_validation_split()
    return fit, {"val": val}, "val"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "eval"])
    ap.add_argument("--protocol", choices=["internal", "frozen"],
                    default="internal")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--data-dir", type=Path, default=Path("results/mpc"))
    args = ap.parse_args()

    data = args.data_dir / f"mpc_arrays_{args.protocol}.npz"
    emb = args.data_dir / f"mpc_embeddings_{args.protocol}.npz"

    corpus, query_sets, eval_set = load_protocol(args.protocol)
    print(f"protocol {args.protocol}: corpus {len(corpus)}, "
          + ", ".join(f"{n} {len(c)}" for n, c in query_sets.items()))

    if args.stage == "prep":
        from src.evaluation.mpc_data import export_mpc_arrays

        exclude = ({tc.reduced_formula for tc in query_sets["es"]}
                   if args.protocol == "frozen" else None)
        meta = export_mpc_arrays(corpus, query_sets, data,
                                 train_exclude=exclude)
        print(f"wrote {data} ({meta['n_elements']} elements, "
              f"{meta['n_precursors']} precursors)")
        print("next: .venv-rr/bin/python scripts/rr_train_mpc.py "
              f"--data {data} --rr-repo <Retrieval-Retro clone> --out {emb}"
              + (" --val-set es" if args.protocol == "frozen" else ""))
        return

    import numpy as np

    from src.evaluation.baselines import ElementJaccardRetriever
    from src.evaluation.mpc_retriever import PrecomputedEmbeddingRetriever
    from src.evaluation.statistics import holm_correction, wilcoxon_table
    from src.evaluation.transferability import evaluate_transferability

    blobs = np.load(emb)
    queries = query_sets[eval_set]
    q_emb = {tc.reduced_formula: e
             for tc, e in zip(queries, blobs[f"{eval_set}_emb"])}
    retrievers = {
        "Element Jaccard": ElementJaccardRetriever(corpus=corpus),
        "RR-MPC": PrecomputedEmbeddingRetriever(
            corpus, blobs["train_emb"], q_emb
        ),
    }
    results = evaluate_transferability(retrievers, queries, corpus, k=args.k)

    scores = {name: results[name]["per_query"] for name in retrievers}
    raw = {f"{a}|{b}": v for (a, b), v in wilcoxon_table(scores).items()}
    holm = holm_correction({k_: v["p_value"] for k_, v in raw.items()})
    results["_pairwise"] = {
        k_: {**v, "p_holm": holm[k_]["p_holm"],
             "significant_holm": holm[k_]["significant"]}
        for k_, v in raw.items()
    }
    results["_config"].update({
        "protocol": args.protocol,
        "model": "Retrieval-Retro MPC retriever (NeurIPS 2024), trained "
                 "on our corpus via scripts/rr_train_mpc.py",
        "best_val_recall10": float(blobs["_best_val_recall10"][0]),
    })

    out = Path(f"results/mpc_baseline_{args.protocol}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\n{'=' * 60}")
    for name in retrievers:
        r = results[name]
        sg = r["per_route"].get("sol-gel", {})
        print(f"{name:16s} fSRO@{args.k}={r['mean']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"sol-gel={sg.get('mean', float('nan')):.3f}  "
              f"regret={r['mean_regret']:.3f}")
    print(f"{'Oracle':16s} fSRO@{args.k}={results['oracle']['mean']:.3f}")
    for pair, v in results["_pairwise"].items():
        print(f"  {pair}: p_holm={v['p_holm']:.4f} sig={v['significant_holm']}")
    print(f"\nresults saved to {out}")


if __name__ == "__main__":
    main()
