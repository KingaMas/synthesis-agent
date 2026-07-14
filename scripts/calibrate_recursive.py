#!/usr/bin/env python
"""
Calibrate RecursiveSynthesisSearch confidence thresholds.

Context: the confidence bandwidth changed from exp(-d/3) to exp(-d/0.5)
(ECE-calibrated), which shrank all confidence values; the recursive-search
defaults (min_confidence=0.7, confidence_decay=0.85) were tuned for the old
scale and prune every neighbor under the new one. This script grid-searches
replacement thresholds against recipe-transfer quality (SRO).

Usage
-----
    PYTHONPATH=. python scripts/calibrate_recursive.py [--max-cases 200]

No API keys required — reads local assets only.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search recursive-search confidence thresholds"
    )
    parser.add_argument("--max-cases", type=int, default=200,
                        help="Test cases to draw the validation split from")
    args = parser.parse_args()

    from src.embedding import InputType
    from src.evaluation.confidence_calibration import grid_search_recursive_params
    from src.evaluation.test_set_builder import build_retrieval_corpus, build_test_set
    from src.search_api import SearchAPI

    print("Building test set and corpus ...")
    test_cases = build_test_set()[: args.max_cases]
    corpus = build_retrieval_corpus()
    print(f"  {len(test_cases)} cases ({len(test_cases) // 5} validation), "
          f"{len(corpus)} corpus materials")

    print("Loading composition SearchAPI ...")
    search_api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=100)

    print("\nGrid search (score = mean best-SRO among reachable neighbors):")
    best = grid_search_recursive_params(
        test_cases, search_api, corpus=corpus, verbose=True
    )

    print(f"\nBest parameters: {best}")
    print(
        "\nApply by updating defaults in:\n"
        "  src/recursive_synthesis.py  RecursiveSynthesisSearch.__init__\n"
        "  sky/core/synthesis_agent.py recursive_synthesis_search tool"
    )


if __name__ == "__main__":
    main()
