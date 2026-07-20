#!/usr/bin/env python
"""
M3 train-internal temporal validation (T4 method ladder, final rung).

Protocol identical to M1/M2: fit <2014, validate 2014-2015, frozen 2016
test split NOT consumed. Two stages around the torch-side trainer:

    PYTHONPATH=. python scripts/run_m3_validation.py prep
    .venv-rr/bin/python scripts/m3_train_contrastive.py \
        --data results/mpc/m3_arrays_internal.npz \
        --out results/mpc/m3_embeddings_internal.npz
    PYTHONPATH=. python scripts/run_m3_validation.py eval

The eval stage scores M3 against element Jaccard and, when their result
files/embeddings exist, RR-MPC and the M2 per-query scores — one
Wilcoxon+Holm family over the identical validation queries.

Pre-registered success criterion (handoff 2026-07-14): formula-SRO@5
>= 0.45 overall AND >= 0.33 sol-gel, Holm-significant vs element
Jaccard. If M3 fails, the ladder's kill criterion triggers.
"""

import argparse
import json
from pathlib import Path

DATA = Path("results/mpc/m3_arrays_internal.npz")
EMB = Path("results/mpc/m3_embeddings_internal.npz")
MPC_EMB = Path("results/mpc/mpc_embeddings_internal.npz")
M2_RESULTS = Path("results/m2_validation_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "eval"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=Path("results/m3_validation_results.json"))
    args = ap.parse_args()

    from src.evaluation.test_set_builder import load_ladder_validation_split

    fit, val = load_ladder_validation_split()
    print(f"fit corpus {len(fit)}, validation queries {len(val)}")

    if args.stage == "prep":
        from src.evaluation.mpc_data import export_mpc_arrays

        meta = export_mpc_arrays(fit, {"val": val}, DATA, include_magpie=True)
        print(f"wrote {DATA} ({meta['n_elements']} elements, "
              f"{meta['n_precursors']} precursors, Magpie included)")
        print("next: .venv-rr/bin/python scripts/m3_train_contrastive.py "
              f"--data {DATA} --out {EMB}")
        return

    import numpy as np

    from src.evaluation.baselines import ElementJaccardRetriever
    from src.evaluation.mpc_retriever import PrecomputedEmbeddingRetriever
    from src.evaluation.statistics import holm_correction, wilcoxon_table
    from src.evaluation.transferability import evaluate_transferability

    def emb_retriever(path: Path) -> PrecomputedEmbeddingRetriever:
        blobs = np.load(path)
        q_emb = {tc.reduced_formula: e
                 for tc, e in zip(val, blobs["val_emb"])}
        return PrecomputedEmbeddingRetriever(fit, blobs["train_emb"], q_emb)

    retrievers = {
        "Element Jaccard": ElementJaccardRetriever(corpus=fit),
        "M3 Contrastive": emb_retriever(EMB),
    }
    if MPC_EMB.exists():
        retrievers["RR-MPC"] = emb_retriever(MPC_EMB)

    results = evaluate_transferability(retrievers, val, fit, k=args.k)

    scores = {name: results[name]["per_query"] for name in retrievers}
    # Merge M2's per-query scores (same protocol, same query order) into
    # the stats family without re-training it.
    if M2_RESULTS.exists():
        m2 = json.loads(M2_RESULTS.read_text())
        if m2["_config"]["query_formulas"] == results["_config"]["query_formulas"]:
            m2_name = next(n for n in m2 if n.startswith("M2 GBRank l31"))
            scores["M2 GBRank"] = m2[m2_name]["per_query"]
        else:
            print("WARNING: M2 results use a different query list; skipped")

    raw = {f"{a}|{b}": v for (a, b), v in wilcoxon_table(scores).items()}
    holm = holm_correction({k_: v["p_value"] for k_, v in raw.items()})
    results["_pairwise"] = {
        k_: {**v, "p_holm": holm[k_]["p_holm"],
             "significant_holm": holm[k_]["significant"]}
        for k_, v in raw.items()
    }
    results["_config"].update({
        "protocol": "fit <2014, validate 2014-2015 "
                    "(train-internal; frozen 2016 test split NOT consumed)",
        "model": "M3 contrastive composition encoder "
                 "(scripts/m3_train_contrastive.py, .venv-rr)",
        "success_criterion": (
            "formula-SRO@5 >= 0.45 overall AND >= 0.33 sol-gel, "
            "Holm-significant vs Element Jaccard"
        ),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\n{'=' * 70}")
    for name in retrievers:
        r = results[name]
        sg = r["per_route"].get("sol-gel", {})
        print(f"{name:18s} fSRO@{args.k}={r['mean']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"sol-gel={sg.get('mean', float('nan')):.3f}  "
              f"regret={r['mean_regret']:.3f}")
    o = results["oracle"]
    print(f"{'Oracle':18s} fSRO@{args.k}={o['mean']:.3f}  "
          f"sol-gel={o['per_route'].get('sol-gel', {}).get('mean', float('nan')):.3f}")
    for pair, v in results["_pairwise"].items():
        print(f"  {pair}: p_holm={v['p_holm']:.4f} sig={v['significant_holm']}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
