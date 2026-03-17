#!/usr/bin/env python
"""
Run the full retrieval benchmark and produce Table 1.

Evaluates MAGPIE (SKY) against four baselines on the synthesis recipe
test set.  All retrievers search the same full recipe corpus (~25k
unique materials).  The 400 stratified test cases serve as queries.

Usage
-----
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --max-cases 200
    python scripts/run_benchmark.py --output results/benchmark.json

Environment
-----------
No API key required — reads local assets only:
    assets/embedding/mp_dataset_composition_magpie.h5
    assets/mp_synthesis_recipes.json.gz
"""

import argparse
import json
from pathlib import Path

# Run from repo root: PYTHONPATH=. .venv/bin/python3 scripts/run_benchmark.py


def main():
    parser = argparse.ArgumentParser(description="Run SKY retrieval benchmark (Table 1)")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit test set size (default: all, ~400 cases)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_results.json"),
        help="Where to save JSON results",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="k values for retrieval metrics (default: 1 3 5 10 20)",
    )
    args = parser.parse_args()

    from src.evaluation.test_set_builder import build_test_set, build_retrieval_corpus
    from src.evaluation.benchmark import RetrievalBenchmark
    from src.evaluation.baselines import (
        RandomRetriever,
        ElementJaccardRetriever,
        StoichiometricVectorRetriever,
        FormulaTFIDFRetriever,
    )
    from src.evaluation.sky_retriever import SKYRetriever
    from src.evaluation.statistics import format_results_table, bootstrap_ci
    from src.search_api import SearchAPI
    from src.embedding import InputType

    # ------------------------------------------------------------------ #
    # Build test queries (stratified sample)
    # ------------------------------------------------------------------ #
    print("Building test set (query cases) ...")
    test_cases = build_test_set()
    if args.max_cases:
        test_cases = test_cases[: args.max_cases]
    print(f"  {len(test_cases)} query cases")
    by_method: dict[str, int] = {}
    for tc in test_cases:
        by_method[tc.synthesis_method] = by_method.get(tc.synthesis_method, 0) + 1
    for method, count in sorted(by_method.items()):
        print(f"    {method:20s}: {count}")

    # ------------------------------------------------------------------ #
    # Build full retrieval corpus (all unique formulas in recipe DB)
    # ------------------------------------------------------------------ #
    print("\nBuilding full retrieval corpus ...")
    full_corpus = build_retrieval_corpus()
    print(f"  {len(full_corpus)} unique recipe materials in corpus")
    by_method_corpus: dict[str, int] = {}
    for tc in full_corpus:
        by_method_corpus[tc.synthesis_method] = by_method_corpus.get(tc.synthesis_method, 0) + 1
    for method, count in sorted(by_method_corpus.items()):
        print(f"    {method:20s}: {count}")

    # ------------------------------------------------------------------ #
    # Setup benchmark
    # ------------------------------------------------------------------ #
    benchmark = RetrievalBenchmark(
        test_cases=test_cases,
        k_values=args.k,
        verbose=True,
    )
    all_results = {}

    # ------------------------------------------------------------------ #
    # MAGPIE (SKY): queries full H5, looks up in full recipe corpus
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Running MAGPIE (SKY) retriever ...")
    api = SearchAPI(
        input_type=InputType.COMPOSITION,
        max_neighbors=max(args.k) * 20 + 50,
    )
    sky = SKYRetriever(api, corpus=full_corpus, fetch_factor=20)
    all_results["MAGPIE (SKY)"] = benchmark.evaluate(sky, "MAGPIE (SKY)")

    # ------------------------------------------------------------------ #
    # Baselines: all search the same full_corpus
    # ------------------------------------------------------------------ #
    for name, cls in [
        ("Random", RandomRetriever),
        ("Element Jaccard", ElementJaccardRetriever),
        ("Stoich Vector", StoichiometricVectorRetriever),
        ("TF-IDF", FormulaTFIDFRetriever),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Running {name} (corpus size: {len(full_corpus)}) ...")
        if name == "Random":
            ret = cls(corpus=full_corpus)
        else:
            ret = cls(corpus=full_corpus)
        all_results[name] = benchmark.evaluate(ret, name)

    # ------------------------------------------------------------------ #
    # Print Table 1
    # ------------------------------------------------------------------ #
    for k in args.k:
        if k != 5 and k != max(args.k):
            continue
        print(f"\n{'=' * 60}")
        print(f"TABLE 1  —  SRO@{k}  (n={len(test_cases)} queries, corpus={len(full_corpus)})")
        print(format_results_table(all_results, metric=f"SRO@{k}", k=k))

    print(f"\n{'=' * 60}")
    print("FULL SUMMARY")
    for name, res in all_results.items():
        print(res.summary_table())
        print()

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable: dict = {}
    for name, res in all_results.items():
        per_k: dict = {}
        for k in args.k:
            mean_sro, lo_sro, hi_sro = bootstrap_ci(res.per_query_sro.get(k, []))
            mean_mcr, lo_mcr, hi_mcr = bootstrap_ci(res.per_query_mcr.get(k, []))
            mean_ndcg, lo_ndcg, hi_ndcg = bootstrap_ci(res.per_query_ndcg.get(k, []))
            per_k[str(k)] = {
                "sro": {"mean": mean_sro, "ci_lo": lo_sro, "ci_hi": hi_sro},
                "mcr": {"mean": mean_mcr, "ci_lo": lo_mcr, "ci_hi": hi_mcr},
                "ndcg": {"mean": mean_ndcg, "ci_lo": lo_ndcg, "ci_hi": hi_ndcg},
            }
        mrr_mean, mrr_lo, mrr_hi = bootstrap_ci(res.per_query_mrr)
        serializable[name] = {
            "n_queries": res.n_queries,
            "corpus_size": len(full_corpus),
            "per_k": per_k,
            "mrr": {"mean": mrr_mean, "ci_lo": mrr_lo, "ci_hi": mrr_hi},
        }

    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("Regenerate Figure 3:  python paper_figures/fig3_retrieval_curves.py")


if __name__ == "__main__":
    main()
