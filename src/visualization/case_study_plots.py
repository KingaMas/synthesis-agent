"""
Per-case-study figure generation helpers.

Used by the three case study notebooks to produce consistent, paper-ready plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_similarity_scores(
    material_names: list[str],
    sro_scores: list[float],
    confidence_scores: list[float],
    title: str = "Synthesis Similarity Scores",
    output_path: Optional[Path] = None,
) -> Path:
    """Horizontal bar chart of SRO and confidence scores for a set of materials.

    Args:
        material_names:   Y-axis labels (reduced formulas).
        sro_scores:       SRO similarity to the query.
        confidence_scores: KNN confidence scores.
        title:            Plot title.
        output_path:      Where to save PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(material_names) * 0.6)))
    y = np.arange(len(material_names))

    axes[0].barh(y, sro_scores, color="#4DBBD5", alpha=0.85)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(material_names, fontsize=10)
    axes[0].set_xlabel("SRO (Jaccard precursor overlap)", fontsize=11)
    axes[0].set_title("Synthesis Recipe Overlap", fontsize=12)
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="SRO=0.3 threshold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(y, confidence_scores, color="#E64B35", alpha=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([""] * len(material_names))
    axes[1].set_xlabel("Embedding confidence score", fontsize=11)
    axes[1].set_title("Retrieval Confidence", fontsize=12)
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path is None:
        safe_title = title.replace(" ", "_").replace("/", "_")[:40]
        output_path = Path("paper_figures") / f"{safe_title}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")
    return output_path


def plot_synthesis_parameter_comparison(
    materials: list[str],
    temperatures: list[Optional[float]],
    methods: list[str],
    title: str = "Synthesis Parameter Comparison",
    output_path: Optional[Path] = None,
) -> Path:
    """Scatter-style chart comparing synthesis temperatures and methods.

    Args:
        materials:     Material labels.
        temperatures:  Synthesis temperatures in °C (None = unknown).
        methods:       Synthesis method family for each material.
        title:         Plot title.
        output_path:   Where to save PNG.
    """
    from src.evaluation.test_set_builder import METHOD_FAMILIES
    from src.visualization.embedding_viz import METHOD_COLOURS

    method_list = list(METHOD_FAMILIES.keys())
    method_idx = {m: i for i, m in enumerate(method_list)}

    y_pos = [method_idx.get(m, len(method_list) - 1) for m in methods]
    colours = [METHOD_COLOURS.get(m, "#B2B2B2") for m in methods]
    sizes = [200 if t is not None else 50 for t in temperatures]
    x_vals = [t if t is not None else 500.0 for t in temperatures]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(x_vals, y_pos, c=colours, s=sizes, alpha=0.8, edgecolors="k", linewidths=0.5)

    for i, mat in enumerate(materials):
        ax.annotate(mat, (x_vals[i], y_pos[i]), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    ax.set_yticks(range(len(method_list)))
    ax.set_yticklabels([m.replace("-", " ").title() for m in method_list], fontsize=10)
    ax.set_xlabel("Synthesis temperature (°C)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        safe_title = title.replace(" ", "_").replace("/", "_")[:40]
        output_path = Path("paper_figures") / f"{safe_title}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")
    return output_path


def plot_search_tree(
    tree_data: dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Visualise the recursive synthesis search tree (Figure 5).

    Args:
        tree_data: Dict with keys 'target', 'recommendations', 'visited_materials'.
        output_path: Where to save PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    target = tree_data.get("target", "Target")
    recs = tree_data.get("recommendations", [])

    # Root node
    ax.scatter([0], [0], s=600, c="#E64B35", zorder=5)
    ax.annotate(target, (0, 0), textcoords="offset points", xytext=(10, 5),
                fontsize=11, fontweight="bold")

    # Recommendation nodes
    n = len(recs)
    if n > 0:
        angles = np.linspace(-np.pi / 3, np.pi / 3, n)
        r = 2.0
        for i, (rec, angle) in enumerate(zip(recs, angles)):
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            conf = rec.get("confidence", 0.5)
            path_len = rec.get("path_length", 1)
            size = 200 + 300 * conf
            color = plt.cm.RdYlGn(conf)
            ax.scatter([x], [y], s=size, c=[color], zorder=5, alpha=0.85)
            ax.plot([0, x], [0, y], "k-", alpha=0.3, linewidth=1 + path_len)
            ax.annotate(
                f"{rec.get('source_material', '')} ({conf:.2f})",
                (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8,
            )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    ax.set_title(f"Recursive Synthesis Search Tree: {target}", fontsize=13)

    if output_path is None:
        output_path = Path("paper_figures") / "fig5_search_tree.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 5 → {output_path}")
    return output_path
