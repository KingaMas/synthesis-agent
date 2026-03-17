"""
Embedding space visualizations for SKY paper figures.

Figure 1 — UMAP of MP composition embedding space, colored by synthesis method.
Figure 2 — Histogram of nearest-neighbor distances stratified by method agreement.
Figure 3 — Retrieval performance curves (SRO@k, MCR@k, NDCG@k vs k).
Figure 4 — Reliability diagram (confidence vs actual precision).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments

import matplotlib.pyplot as plt
import numpy as np

from src import ASSETS_DIR
from src.evaluation.test_set_builder import METHOD_FAMILIES


# ---------------------------------------------------------------------------
# Colour palette — one colour per synthesis method family
# ---------------------------------------------------------------------------

METHOD_COLOURS = {
    "solid-state":   "#E64B35",
    "hydrothermal":  "#4DBBD5",
    "sol-gel":       "#00A087",
    "combustion":    "#F39B7F",
    "precipitation": "#8491B4",
    "other":         "#B2B2B2",
}


def _load_magpie_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MAGPIE features, material IDs, and formulas from the HDF5 file."""
    import h5py
    h5_file = ASSETS_DIR / "embedding" / "mp_dataset_composition_magpie.h5"
    with h5py.File(h5_file, "r") as f:
        features = f["features"][:]
        material_ids = f["material_ids"][:].astype("str")
        formulas = f["formulas"][:].astype("str")
    return features, material_ids, formulas


# ---------------------------------------------------------------------------
# Figure 1: UMAP embedding space
# ---------------------------------------------------------------------------

def plot_umap_embedding(
    max_points: int = 10_000,
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> Path:
    """Generate Figure 1: UMAP of MAGPIE embedding coloured by synthesis method.

    Args:
        max_points:  Subsample size (UMAP is O(n^2) without approximation).
        output_path: Where to save the PNG (default: paper_figures/fig1_umap_embedding.png).
        seed:        Random seed for reproducibility.

    Returns:
        Path to the saved figure.
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is required: pip install umap-learn") from e

    from src.evaluation.test_set_builder import (
        load_recipes,
        classify_synthesis_method,
        _extract_precursor_elements,
    )
    from pymatgen.core import Composition

    features, material_ids, formulas = _load_magpie_embeddings()

    # Build method label for each formula via recipe DB
    recipes = load_recipes()
    formula_to_method: dict[str, str] = {}
    for recipe in recipes:
        target = recipe.get("target_formula", "")
        if not target:
            continue
        try:
            reduced = Composition(target).reduced_formula
        except Exception:
            continue
        if reduced not in formula_to_method:
            text = recipe.get("synthesis_type", "") + " " + recipe.get("paragraph_string", "")
            formula_to_method[reduced] = classify_synthesis_method(text)

    labels: list[str] = []
    for f in formulas:
        try:
            reduced = Composition(f).reduced_formula
        except Exception:
            reduced = f
        labels.append(formula_to_method.get(reduced, "other"))

    # Subsample
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size=min(max_points, len(features)), replace=False)
    X = features[indices]
    y = [labels[i] for i in indices]

    # Standardise
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for method, colour in METHOD_COLOURS.items():
        mask = np.array([m == method for m in y])
        if mask.any():
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=colour,
                label=method.replace("-", " ").title(),
                s=4,
                alpha=0.6,
                rasterized=True,
            )

    ax.set_xlabel("UMAP 1", fontsize=13)
    ax.set_ylabel("UMAP 2", fontsize=13)
    ax.set_title("MAGPIE Composition Embedding Space\n(coloured by synthesis method family)", fontsize=14)
    ax.legend(markerscale=3, title="Synthesis method", fontsize=10, title_fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if output_path is None:
        output_path = Path("paper_figures") / "fig1_umap_embedding.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Figure 2: Nearest-neighbor distance distribution
# ---------------------------------------------------------------------------

def plot_distance_distribution(
    n_samples: int = 5_000,
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> Path:
    """Generate Figure 2: Histogram of KNN distances, same vs different method.

    Args:
        n_samples:   Number of query pairs to sample.
        output_path: Where to save the PNG.
        seed:        Random seed.
    """
    from src.search_api import SearchAPI
    from src.embedding import InputType
    from src.evaluation.test_set_builder import load_recipes, classify_synthesis_method
    from pymatgen.core import Composition

    api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=6)

    recipes = load_recipes()
    formula_to_method: dict[str, str] = {}
    for recipe in recipes:
        t = recipe.get("target_formula", "")
        if not t:
            continue
        try:
            reduced = Composition(t).reduced_formula
        except Exception:
            continue
        if reduced not in formula_to_method:
            text = recipe.get("synthesis_type", "") + " " + recipe.get("paragraph_string", "")
            formula_to_method[reduced] = classify_synthesis_method(text)

    formulas_with_method = list(formula_to_method.items())
    rng = np.random.default_rng(seed)
    sample = [formulas_with_method[i] for i in rng.choice(len(formulas_with_method), size=min(n_samples, len(formulas_with_method)), replace=False)]

    same_distances: list[float] = []
    diff_distances: list[float] = []

    for formula, method in sample:
        try:
            comp = Composition(formula)
            neighbors = api.query(comp, n_neighbors=5)
            for n in neighbors[1:]:  # skip self (rank 0)
                try:
                    n_method = formula_to_method.get(
                        Composition(n.formula).reduced_formula, "other"
                    )
                except Exception:
                    n_method = "other"
                if n_method == method:
                    same_distances.append(n.distance)
                else:
                    diff_distances.append(n.distance)
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(same_distances + diff_distances + [1]), 50)
    ax.hist(same_distances, bins=bins, alpha=0.6, label="Same method", color="#4DBBD5", density=True)
    ax.hist(diff_distances, bins=bins, alpha=0.6, label="Different method", color="#E64B35", density=True)
    ax.set_xlabel("Euclidean distance in MAGPIE embedding space", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("KNN Distance Distribution by Synthesis Method Agreement", fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if output_path is None:
        output_path = Path("paper_figures") / "fig2_distance_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2 → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Figure 3: Retrieval performance curves
# ---------------------------------------------------------------------------

def plot_retrieval_curves(
    results_dict: dict,
    k_values: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate Figure 3: SRO@k, MCR@k, NDCG@k vs k for all methods.

    Args:
        results_dict: Dict[str, BenchmarkResults] from RetrievalBenchmark.evaluate().
        k_values:     k-axis ticks.
        output_path:  Where to save the PNG.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    metrics = ["sro", "mcr", "ndcg"]
    metric_labels = {"sro": "SRO@k", "mcr": "MCR@k", "ndcg": "NDCG@k"}
    colours = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, metric in zip(axes, metrics):
        for i, (name, res) in enumerate(results_dict.items()):
            values = [getattr(res, metric).get(k, float("nan")) for k in k_values]
            ax.plot(k_values, values, marker="o", label=name, color=colours[i % len(colours)])
        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(metric_labels[metric], fontsize=13)
        ax.set_xticks(k_values)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Retrieval Performance vs k", fontsize=14, y=1.01)
    fig.tight_layout()

    if output_path is None:
        output_path = Path("paper_figures") / "fig3_retrieval_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 3 → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Figure 4: Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    calibration_results: list,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate Figure 4: Reliability diagram for confidence calibration.

    Args:
        calibration_results: List of CalibrationResult from calibrate_bandwidth().
        output_path:         Where to save the PNG.
    """
    colours = plt.cm.viridis(np.linspace(0, 1, len(calibration_results)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_diag, ax_ece = axes

    for i, cal in enumerate(calibration_results):
        valid = [(c, a) for c, a, n in zip(cal.bin_confidences, cal.bin_accuracies, cal.bin_counts)
                 if not (a != a) and n > 0]  # filter NaN
        if valid:
            confs, accs = zip(*valid)
            ax_diag.plot(confs, accs, marker="o", label=f"σ={cal.sigma}", color=colours[i], alpha=0.8)

    ax_diag.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax_diag.set_xlabel("Mean predicted confidence", fontsize=12)
    ax_diag.set_ylabel("Empirical precision", fontsize=12)
    ax_diag.set_title("Reliability Diagram", fontsize=13)
    ax_diag.legend(fontsize=9)
    ax_diag.grid(True, alpha=0.3)

    # ECE bar chart
    sigmas = [c.sigma for c in calibration_results]
    eces = [c.ece for c in calibration_results]
    ax_ece.bar(range(len(sigmas)), eces, color=colours, alpha=0.85)
    ax_ece.set_xticks(range(len(sigmas)))
    ax_ece.set_xticklabels([f"σ={s}" for s in sigmas], fontsize=10)
    ax_ece.set_ylabel("Expected Calibration Error (ECE)", fontsize=12)
    ax_ece.set_title("ECE vs Bandwidth σ", fontsize=13)
    ax_ece.grid(True, axis="y", alpha=0.3)
    best_idx = int(np.argmin(eces))
    ax_ece.get_children()[best_idx].set_edgecolor("red")
    ax_ece.get_children()[best_idx].set_linewidth(2)

    fig.tight_layout()

    if output_path is None:
        output_path = Path("paper_figures") / "fig4_calibration.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 4 → {output_path}")
    return output_path
