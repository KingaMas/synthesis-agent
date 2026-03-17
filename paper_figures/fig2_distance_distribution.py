#!/usr/bin/env python
"""
Generate Figure 2: Histogram of nearest-neighbor distances.

Usage:
    python paper_figures/fig2_distance_distribution.py
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2: Distance distribution")
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.visualization.embedding_viz import plot_distance_distribution
    out = plot_distance_distribution(
        n_samples=args.n_samples,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Figure 2 saved to: {out}")


if __name__ == "__main__":
    main()
