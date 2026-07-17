#!/usr/bin/env python
"""
M2 train-internal temporal validation (T4 method ladder).

Same protocol as M1: fit on train-corpus materials first published
before 2014, validate on 2014-2015 materials (n=1042), retrieval corpus
for validation queries = fit corpus. The frozen 2016 temporal TEST split
is never touched here; it is spent only if a configuration passes the
pre-registered criterion on validation (formula-SRO@5 >= 0.45 overall
AND >= 0.33 sol-gel, Holm-significant vs element Jaccard).

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/run_m2_validation.py \
        [--max-val N] [--grid]
"""

import argparse
import json
from pathlib import Path

FIT_BEFORE = 2014
VAL_BEFORE = 2016


def main():
    parser = argparse.ArgumentParser(description="M2 internal validation")
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--grid", action="store_true",
                        help="sweep a small hyperparameter grid")
    parser.add_argument("--output", type=Path,
                        default=Path("results/m2_validation_results.json"))
    args = parser.parse_args()

    from src import PROJECT_ROOT
    from src.evaluation.baselines import ElementJaccardRetriever
    from src.evaluation.form_rerank import FormCooccurrenceModel, FormRerankRetriever
    from src.evaluation.pairwise_rank import (
        GradientBoostedRanker,
        PairFeatureExtractor,
        PairwiseRankRetriever,
    )
    from src.evaluation.statistics import holm_correction, wilcoxon_table
    from src.evaluation.test_set_builder import load_temporal_split
    from src.evaluation.transferability import evaluate_transferability

    print("Loading frozen temporal split (test side ignored) ...")
    train, _test_unused = load_temporal_split()
    years = json.loads(
        (PROJECT_ROOT / "results" / "doi_years.json").read_text()
    )

    def year_of(tc):
        return years.get(tc.raw_recipe.get("doi") or "")

    fit = [tc for tc in train if year_of(tc) is not None
           and year_of(tc) < FIT_BEFORE]
    val = [tc for tc in train if year_of(tc) is not None
           and FIT_BEFORE <= year_of(tc) < VAL_BEFORE]
    if args.max_val:
        val = val[: args.max_val]
    print(f"  fit corpus {len(fit)} (<{FIT_BEFORE}), "
          f"validation queries {len(val)} ({FIT_BEFORE}-{VAL_BEFORE - 1})")

    print("Fitting feature extractor on fit corpus ...")
    extractor = PairFeatureExtractor(fit)
    jaccard = ElementJaccardRetriever(corpus=fit)

    configs = [dict(num_leaves=31, n_estimators=300, learning_rate=0.05)]
    if args.grid:
        configs = [
            dict(num_leaves=31, n_estimators=300, learning_rate=0.05),
            dict(num_leaves=15, n_estimators=500, learning_rate=0.03),
            dict(num_leaves=63, n_estimators=200, learning_rate=0.05),
        ]

    retrievers = {
        "Element Jaccard": jaccard,
        "M1 Form Rerank": FormRerankRetriever(
            jaccard, FormCooccurrenceModel(fit), alpha=0.5
        ),
    }
    rankers = {}
    for cfg in configs:
        name = (f"M2 GBRank l{cfg['num_leaves']}"
                f"n{cfg['n_estimators']}lr{cfg['learning_rate']}")
        print(f"Training {name} ...")
        ranker = GradientBoostedRanker(extractor, **cfg).fit(fit, jaccard)
        rankers[name] = ranker
        retrievers[name] = PairwiseRankRetriever(jaccard, ranker)

    results = evaluate_transferability(retrievers, val, fit, k=args.k)

    scores = {name: results[name]["per_query"] for name in retrievers}
    raw = {f"{a}|{b}": v for (a, b), v in wilcoxon_table(scores).items()}
    holm = holm_correction({k_: v["p_value"] for k_, v in raw.items()})
    results["_pairwise"] = {
        k_: {**v, "p_holm": holm[k_]["p_holm"],
             "significant_holm": holm[k_]["significant"]}
        for k_, v in raw.items()
    }
    results["_config"].update({
        "protocol": f"fit <{FIT_BEFORE}, validate {FIT_BEFORE}-{VAL_BEFORE - 1}"
                    " (train-internal; frozen 2016 test split NOT consumed)",
        "m2_configs": configs,
        "success_criterion": (
            "formula-SRO@5 >= 0.45 overall AND >= 0.33 sol-gel, "
            "Holm-significant vs Element Jaccard"
        ),
    })
    results["_feature_importances"] = {
        name: r.feature_importances() for name, r in rankers.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\n{'=' * 70}")
    for name in retrievers:
        r = results[name]
        sg = r["per_route"].get("sol-gel", {})
        print(f"{name:28s} fSRO@{args.k}={r['mean']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"sol-gel={sg.get('mean', float('nan')):.3f}  "
              f"regret={r['mean_regret']:.3f}")
    print(f"{'Oracle':28s} fSRO@{args.k}={results['oracle']['mean']:.3f}  "
          f"sol-gel={results['oracle']['per_route'].get('sol-gel', {}).get('mean', float('nan')):.3f}")
    for pair, v in results["_pairwise"].items():
        print(f"  {pair}: p_holm={v['p_holm']:.4f} sig={v['significant_holm']}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
