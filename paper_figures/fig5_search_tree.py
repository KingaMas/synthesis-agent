#!/usr/bin/env python
"""
Generate Figure 5: Recursive synthesis search tree visualization.

Demonstrates the recursive search algorithm for a rare perovskite material.

Usage:
    python paper_figures/fig5_search_tree.py
    python paper_figures/fig5_search_tree.py --formula Ba2FeMoO6 --output figs/fig5.png

Requires MP_API_KEY environment variable.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5: Search tree")
    parser.add_argument("--formula", default="Ba2FeMoO6")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from src.agent import SynthesisAgent
    from src.recursive_synthesis import RecursiveSynthesisSearch
    from src.visualization.case_study_plots import plot_search_tree

    print(f"Running recursive search for {args.formula}...")
    agent = SynthesisAgent()
    searcher = RecursiveSynthesisSearch(
        synthesis_agent=agent,
        max_depth=args.max_depth,
        min_confidence=0.7,
        verbose=True,
    )
    results = searcher.search(args.formula, n_initial_neighbors=10)

    print(f"\nSearch complete. Status: {results['status']}")
    print(f"Visited materials: {results['visited_materials']}")
    print(f"Recipes found: {results.get('unique_materials_with_recipes', 0)}")

    out = plot_search_tree(results, output_path=args.output)
    print(f"Figure 5 saved to: {out}")


if __name__ == "__main__":
    main()
