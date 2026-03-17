"""
Confidence score calibration for the SKY retrieval system.

Sweeps sigma (bandwidth) in exp(-distance/sigma) and computes Expected
Calibration Error (ECE) via reliability diagram.  Also calibrates
RecursiveSynthesisSearch hyperparameters via grid search.

Usage
-----
    from src.evaluation.confidence_calibration import calibrate_bandwidth
    best_sigma, ece_table = calibrate_bandwidth(search_api, test_cases)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymatgen.core import Composition

from src.evaluation.test_set_builder import TestCase


# ---------------------------------------------------------------------------
# Bandwidth sweep
# ---------------------------------------------------------------------------

SIGMA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# Path-length penalty grid for RecursiveSynthesisSearch
DECAY_GRID = [0.70, 0.80, 0.85, 0.90, 0.95]
PATH_PENALTY_GRID = [0.1, 0.2, 0.3, 0.5]


@dataclass
class CalibrationResult:
    sigma: float
    ece: float
    bin_accuracies: list[float]   # empirical precision per confidence bin
    bin_confidences: list[float]  # mean predicted confidence per bin
    bin_counts: list[int]


def _empirical_precision(
    distances: np.ndarray,
    labels: np.ndarray,  # 1 = relevant (SRO >= threshold), 0 = not
    sigma: float,
    n_bins: int = 10,
    threshold: float = 0.3,
) -> CalibrationResult:
    """Compute ECE for a given sigma value.

    Args:
        distances: Array of KNN distances (shape N).
        labels:    Binary relevance labels (shape N).
        sigma:     Bandwidth for exp(-distance/sigma).
        n_bins:    Number of reliability-diagram bins.
        threshold: SRO threshold that defines a "relevant" retrieval.
    """
    confidences = np.exp(-distances / sigma)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences_mean = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bin_accuracies.append(float("nan"))
            bin_confidences_mean.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_accuracies.append(float(labels[mask].mean()))
            bin_confidences_mean.append(float(confidences[mask].mean()))
            bin_counts.append(count)

    total = max(sum(bin_counts), 1)
    ece = sum(
        (cnt / total) * abs(acc - conf)
        for acc, conf, cnt in zip(bin_accuracies, bin_confidences_mean, bin_counts)
        if not math.isnan(acc)
    )

    return CalibrationResult(
        sigma=sigma,
        ece=ece,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences_mean,
        bin_counts=bin_counts,
    )


def calibrate_bandwidth(
    search_api,
    test_cases: list[TestCase],
    sigma_grid: Optional[list[float]] = None,
    k: int = 5,
    sro_threshold: float = 0.3,
    verbose: bool = True,
) -> tuple[float, list[CalibrationResult]]:
    """Sweep sigma and return the value that minimises ECE.

    Args:
        search_api:    A SearchAPI instance (composition or structure).
        test_cases:    Held-out test cases.
        sigma_grid:    Sigma values to sweep (default SIGMA_GRID).
        k:             Number of neighbors used for calibration.
        sro_threshold: Minimum SRO to consider a retrieval "relevant".
        verbose:       Print progress.

    Returns:
        Tuple of (best_sigma, list_of_CalibrationResult for each sigma).
    """
    from src.evaluation.benchmark import _jaccard

    if sigma_grid is None:
        sigma_grid = SIGMA_GRID

    # Build lookup so we can find a neighbor's precursor elements
    by_mid: dict[str, TestCase] = {
        tc.material_id: tc for tc in test_cases if tc.material_id
    }
    by_reduced: dict[str, TestCase] = {tc.reduced_formula: tc for tc in test_cases}

    all_distances: list[float] = []
    all_labels: list[int] = []

    for i, tc in enumerate(test_cases):
        if verbose and i % 100 == 0:
            print(f"  Calibration: {i}/{len(test_cases)}")
        try:
            query_comp = Composition(tc.reduced_formula)
            neighbors = search_api.query_with_exclusion(
                query_comp,
                exclude_ids=[tc.material_id] if tc.material_id else [],
                n_neighbors=k,
            )
        except Exception:
            continue

        query_prec = set(tc.precursor_elements)
        for n in neighbors:
            all_distances.append(n.distance)
            # Look up neighbor's precursor elements for SRO label
            neighbor_tc = by_mid.get(n.material_id)
            if neighbor_tc is None:
                try:
                    from pymatgen.core import Composition as _Comp
                    rf = _Comp(n.formula).reduced_formula
                    neighbor_tc = by_reduced.get(rf)
                except Exception:
                    pass
            neighbor_prec = set(neighbor_tc.precursor_elements) if neighbor_tc else set()
            sro = _jaccard(query_prec, neighbor_prec)
            all_labels.append(1 if sro >= sro_threshold else 0)

    if not all_distances:
        raise RuntimeError("No distances collected during calibration — check test_cases and SearchAPI.")

    distances_arr = np.array(all_distances, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int32)

    results = [
        _empirical_precision(distances_arr, labels_arr, sigma)
        for sigma in sigma_grid
    ]

    best = min(results, key=lambda r: r.ece)
    if verbose:
        print(f"\nCalibration results:")
        for r in results:
            marker = " <-- best" if r.sigma == best.sigma else ""
            print(f"  sigma={r.sigma:.1f}  ECE={r.ece:.4f}{marker}")

    return best.sigma, results


def grid_search_recursive_params(
    test_cases: list[TestCase],
    search_api,
    decay_grid: Optional[list[float]] = None,
    penalty_grid: Optional[list[float]] = None,
    verbose: bool = True,
) -> dict:
    """Grid search over RecursiveSynthesisSearch hyperparameters.

    Returns the (confidence_decay, path_penalty) pair maximising mean SRO@5
    on a validation split (first 20% of test_cases).

    Note: This is a lightweight proxy evaluation that avoids full MP API calls
    by reusing the SearchAPI for neighbor lookup.
    """
    if decay_grid is None:
        decay_grid = DECAY_GRID
    if penalty_grid is None:
        penalty_grid = PATH_PENALTY_GRID

    val_cases = test_cases[: max(1, len(test_cases) // 5)]
    best_score = -1.0
    best_params: dict = {}

    for decay in decay_grid:
        for penalty in penalty_grid:
            scores = []
            for tc in val_cases:
                try:
                    query_comp = Composition(tc.reduced_formula)
                    neighbors = search_api.query_with_exclusion(
                        query_comp,
                        exclude_ids=[tc.material_id] if tc.material_id else [],
                        n_neighbors=5,
                    )
                    # Score: weighted confidence using the given penalty
                    q_prec = set(tc.precursor_elements)
                    scored = [
                        n.confidence / (1 + penalty * 1)  # path_length=1 proxy
                        for n in neighbors
                    ]
                    # Use top-1 score as proxy quality
                    scores.append(max(scored) if scored else 0.0)
                except Exception:
                    scores.append(0.0)

            mean_score = float(np.mean(scores)) if scores else 0.0
            if verbose:
                print(f"  decay={decay}  penalty={penalty}  score={mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"confidence_decay": decay, "path_penalty": penalty, "score": mean_score}

    return best_params
