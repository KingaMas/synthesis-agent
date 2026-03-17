#!/usr/bin/env python
"""
Generate Figure 3: Retrieval performance curves (SRO, MCR, NDCG vs k).

Runs the full benchmark for MAGPIE + all 4 baselines, then plots curves.

Usage:
    python paper_figures/fig3_retrieval_curves.py
    python paper_figures/fig3_retrieval_curves.py --max-cases 500 --output figs/fig3.png
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 3: Retrieval curves")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from src.evaluation.test_set_builder import build_test_set
    from src.evaluation.benchmark import RetrievalBenchmark
    from src.evaluation.baselines import (
        RandomRetriever,
        ElementJaccardRetriever,
        StoichiometricVectorRetriever,
        FormulaTFIDFRetriever,
    )
    from src.visualization.embedding_viz import plot_retrieval_curves
    from src.search_api import SearchAPI
    from src.embedding import InputType
    from pymatgen.core import Composition

    k_values = [1, 3, 5, 10, 20]
    print("Building test set...")
    test_cases = build_test_set()
    if args.max_cases:
        test_cases = test_cases[: args.max_cases]
    print(f"Test set size: {len(test_cases)}")

    lookup = {tc.reduced_formula: tc for tc in test_cases}
    benchmark = RetrievalBenchmark(test_cases=test_cases, k_values=k_values, verbose=True)

    all_results = {}

    # MAGPIE
    print("\n--- Running MAGPIE ---")
    api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=max(k_values) + 10)

    class MAGPIERetriever:
        def retrieve(self, query, k):
            try:
                comp = Composition(query.reduced_formula)
                neighbors = api.query_with_exclusion(
                    comp,
                    exclude_ids=[query.material_id] if query.material_id else [],
                    n_neighbors=k,
                )
                return [lookup[n.formula] for n in neighbors if n.formula in lookup][:k]
            except Exception:
                return []

    all_results["MAGPIE"] = benchmark.evaluate(MAGPIERetriever(), "MAGPIE")

    # Baselines
    for name, cls in [
        ("Random", RandomRetriever),
        ("Element Jaccard", ElementJaccardRetriever),
        ("Stoich Vector", StoichiometricVectorRetriever),
        ("TF-IDF", FormulaTFIDFRetriever),
    ]:
        print(f"\n--- Running {name} ---")
        if name == "Random":
            ret = cls(corpus=test_cases)
        else:
            ret = cls(corpus=test_cases)
        all_results[name] = benchmark.evaluate(ret, name)

    out = plot_retrieval_curves(all_results, k_values=k_values, output_path=args.output)
    print(f"\nFigure 3 saved to: {out}")


if __name__ == "__main__":
    main()
