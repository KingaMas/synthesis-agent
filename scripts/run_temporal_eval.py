#!/usr/bin/env python
"""
Evaluate retrievers on the frozen temporal split (T4 method ladder).

Train corpus: materials first published before 2016 (n=5,903).
Test queries: materials first published 2016+ (n=1,595) — the realistic
"new material" scenario. Metric: formula-SRO@5 with oracle calibration
and per-route stratification; paired Wilcoxon vs the element-Jaccard
baseline, Holm-corrected.

Pre-registered success criterion for the M-ladder (handoff 2026-07-14):
formula-SRO@5 >= 0.45 overall AND >= 0.33 on sol-gel queries, with a
Holm-significant paired improvement over element Jaccard.

Usage
-----
    PYTHONPATH=. python scripts/run_temporal_eval.py [--max-cases N]
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Temporal-split retrieval eval")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="M1 rerank weight (a priori default 0.5)")
    parser.add_argument("--output", type=Path,
                        default=Path("results/temporal_eval_results.json"))
    args = parser.parse_args()

    from src.evaluation.baselines import (
        ElementJaccardRetriever,
        StoichiometricVectorRetriever,
    )
    from src.evaluation.form_rerank import FormCooccurrenceModel, FormRerankRetriever
    from src.evaluation.statistics import holm_correction, wilcoxon_table
    from src.evaluation.test_set_builder import load_temporal_split
    from src.evaluation.transferability import evaluate_transferability

    print("Loading frozen temporal split ...")
    train, test = load_temporal_split()
    if args.max_cases:
        test = test[: args.max_cases]
    print(f"  train corpus {len(train)}, test queries {len(test)}")

    print("Fitting M1 form co-occurrence model on train corpus ...")
    model = FormCooccurrenceModel(train)
    jaccard = ElementJaccardRetriever(corpus=train)

    retrievers = {
        "Element Jaccard": jaccard,
        "Stoich Vector": StoichiometricVectorRetriever(corpus=train),
        "M1 Form Rerank": FormRerankRetriever(jaccard, model, alpha=args.alpha),
    }

    results = evaluate_transferability(retrievers, test, train, k=args.k)

    # Paired stats vs element Jaccard, Holm-corrected over the family
    scores = {name: results[name]["per_query"] for name in retrievers}
    raw = {f"{a}|{b}": v for (a, b), v in wilcoxon_table(scores).items()}
    holm = holm_correction({k_: v["p_value"] for k_, v in raw.items()})
    results["_pairwise"] = {
        k_: {**v, "p_holm": holm[k_]["p_holm"],
             "significant_holm": holm[k_]["significant"]}
        for k_, v in raw.items()
    }
    results["_config"]["split"] = "results/temporal_split_2016.json"
    results["_config"]["m1_alpha"] = args.alpha
    results["_config"]["success_criterion"] = (
        "formula-SRO@5 >= 0.45 overall AND >= 0.33 sol-gel, "
        "Holm-significant vs Element Jaccard"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\n{'=' * 60}")
    for name in retrievers:
        r = results[name]
        sg = r["per_route"].get("sol-gel", {})
        print(f"{name:18s} fSRO@{args.k}={r['mean']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"sol-gel={sg.get('mean', float('nan')):.3f}  "
              f"regret={r['mean_regret']:.3f}")
    o = results["oracle"]
    print(f"{'Oracle':18s} fSRO@{args.k}={o['mean']:.3f}")
    for pair, v in results["_pairwise"].items():
        print(f"  {pair}: p_holm={v['p_holm']:.4f} sig={v['significant_holm']}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
