#!/usr/bin/env python
"""
Run the formula-level recipe-transferability evaluation (Table 1 extension).

Scores all retrievers on formula-SRO@5 (non-circular: Jaccard over
normalized precursor formula sets) against the oracle ceiling, with
per-route stratification. Writes results/transferability_results.json,
including the exact query list used.

Usage
-----
    PYTHONPATH=. python scripts/run_transferability.py [--max-cases N]

No API keys required.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Formula-SRO transferability eval")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("results/transferability_results.json"))
    args = parser.parse_args()

    from src.embedding import InputType
    from src.evaluation.baselines import (
        ElementJaccardRetriever,
        FormulaTFIDFRetriever,
        HybridRRFRetriever,
        RandomRetriever,
        StoichiometricVectorRetriever,
    )
    from src.evaluation.sky_retriever import SKYRetriever
    from src.evaluation.test_set_builder import build_retrieval_corpus, build_test_set
    from src.evaluation.transferability import evaluate_transferability
    from src.search_api import SearchAPI

    print("Building test set and corpus ...")
    test_cases = build_test_set()
    if args.max_cases:
        test_cases = test_cases[: args.max_cases]
    corpus = build_retrieval_corpus()
    print(f"  {len(test_cases)} queries over {len(corpus)} corpus materials")

    api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=args.k * 20 + 50)
    sky = SKYRetriever(api, corpus=corpus, fetch_factor=20)
    stoich = StoichiometricVectorRetriever(corpus=corpus)

    retrievers = {
        "MAGPIE (SKY)": sky,
        "Random": RandomRetriever(corpus=corpus),
        "Element Jaccard": ElementJaccardRetriever(corpus=corpus),
        "Stoich Vector": stoich,
        "TF-IDF": FormulaTFIDFRetriever(corpus=corpus),
        "Hybrid (MAGPIE+Stoich RRF)": HybridRRFRetriever([sky, stoich]),
    }

    results = evaluate_transferability(
        retrievers, test_cases, corpus, k=args.k, verbose=True
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
