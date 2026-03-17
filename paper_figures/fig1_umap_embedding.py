#!/usr/bin/env python
"""
Generate Figure 1: UMAP of the MP composition embedding space.

Usage:
    python paper_figures/fig1_umap_embedding.py
    python paper_figures/fig1_umap_embedding.py --max-points 5000 --output figs/fig1.png

Produces: paper_figures/fig1_umap_embedding.png
Requires: umap-learn (pip install umap-learn)
No API keys required — reads local HDF5 embeddings and synthesis recipe DB.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1: UMAP embedding")
    parser.add_argument("--max-points", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.visualization.embedding_viz import plot_umap_embedding
    out = plot_umap_embedding(
        max_points=args.max_points,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Figure 1 saved to: {out}")


if __name__ == "__main__":
    main()
