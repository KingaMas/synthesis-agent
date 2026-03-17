#!/usr/bin/env python
"""
Generate Figure 4: Reliability diagram + ECE vs sigma.

Usage:
    python paper_figures/fig4_calibration.py
    python paper_figures/fig4_calibration.py --max-cases 500 --output figs/fig4.png
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4: Calibration diagram")
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from src.evaluation.test_set_builder import build_test_set
    from src.evaluation.confidence_calibration import calibrate_bandwidth
    from src.visualization.embedding_viz import plot_reliability_diagram
    from src.search_api import SearchAPI
    from src.embedding import InputType

    print("Building test set...")
    test_cases = build_test_set()
    if args.max_cases:
        test_cases = test_cases[: args.max_cases]
    print(f"Test set size: {len(test_cases)}")

    api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=10)
    best_sigma, cal_results = calibrate_bandwidth(api, test_cases, verbose=True)

    print(f"\nBest sigma = {best_sigma}")

    out = plot_reliability_diagram(cal_results, output_path=args.output)
    print(f"Figure 4 saved to: {out}")


if __name__ == "__main__":
    main()
