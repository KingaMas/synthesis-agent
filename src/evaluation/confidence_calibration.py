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

# Grids for RecursiveSynthesisSearch hyperparameters. The confidence grids
# span low values because sigma=0.5 confidences decay fast: a neighbor at
# scaled distance 1.0 scores exp(-2) ~= 0.135.
MIN_CONFIDENCE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
DECAY_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85]
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
    corpus: Optional[list[TestCase]] = None,
    min_confidence_grid: Optional[list[float]] = None,
    decay_grid: Optional[list[float]] = None,
    n_neighbors: int = 10,
    tolerance: float = 0.005,
    verbose: bool = True,
) -> dict:
    """Grid search over RecursiveSynthesisSearch confidence thresholds.

    Simulates the actual depth-1 prune rule from _recursive_search: a
    neighbor's recipes are reachable iff its confidence clears BOTH the
    child filter (conf >= 1.0 * confidence_decay) and the exploration gate
    (conf >= min_confidence), i.e. conf >= max(decay, min_confidence).

    Score for a parameter pair = mean over validation cases of the best
    SRO among reachable neighbors (0.0 when nothing survives pruning, so
    over-aggressive thresholds are penalised through lost coverage).

    Selection uses a parsimony rule: the score is monotone non-decreasing
    as thresholds loosen (a depth-1 proxy cannot see the exploration cost
    that thresholds control in full recursion), so among pairs scoring
    within *tolerance* of the maximum we return the LARGEST effective
    threshold — the most restrictive setting that loses nothing.

    The path-length penalty in _synthesize_results is NOT searched here:
    a depth-1 simulation gives every candidate the same path length, so
    the penalty is unidentifiable. PATH_PENALTY_GRID is kept for a future
    full-recursion sweep.

    Args:
        test_cases: Query cases (first 20% used as validation split).
        search_api: Composition SearchAPI for neighbor lookup.
        corpus:     Lookup corpus for neighbors' precursor elements
                    (defaults to test_cases; pass build_retrieval_corpus()
                    for accurate SRO labels).
        min_confidence_grid / decay_grid: Values to sweep.
        n_neighbors: Neighbors fetched per query.
        verbose:    Print the sweep table.

    Returns:
        {"min_confidence", "confidence_decay", "score", "coverage"} of the
        best pair.
    """
    from src.evaluation.benchmark import _jaccard

    if min_confidence_grid is None:
        min_confidence_grid = MIN_CONFIDENCE_GRID
    if decay_grid is None:
        decay_grid = DECAY_GRID

    lookup_cases = corpus if corpus is not None else test_cases
    by_reduced: dict[str, TestCase] = {tc.reduced_formula: tc for tc in lookup_cases}

    val_cases = test_cases[: max(1, len(test_cases) // 5)]

    # Collect (confidence, sro) pairs per query ONCE; the sweep is then a
    # cheap threshold filter over cached pairs.
    per_query_pairs: list[list[tuple[float, float]]] = []
    for tc in val_cases:
        pairs: list[tuple[float, float]] = []
        try:
            query_comp = Composition(tc.reduced_formula)
            neighbors = search_api.query_with_exclusion(
                query_comp,
                exclude_ids=[tc.material_id] if tc.material_id else [],
                n_neighbors=n_neighbors,
            )
        except Exception:
            neighbors = []
        q_prec = set(tc.precursor_elements)
        for n in neighbors:
            try:
                rf = Composition(n.formula).reduced_formula
            except Exception:
                continue
            if rf == tc.reduced_formula:
                continue  # self-retrieval (material_id is empty in real data)
            neighbor_tc = by_reduced.get(rf)
            neighbor_prec = set(neighbor_tc.precursor_elements) if neighbor_tc else set()
            pairs.append((n.confidence, _jaccard(q_prec, neighbor_prec)))
        per_query_pairs.append(pairs)

    candidates: list[dict] = []
    for min_conf in min_confidence_grid:
        for decay in decay_grid:
            threshold = max(min_conf, decay)
            scores = []
            explored = 0
            for pairs in per_query_pairs:
                reachable = [sro for conf, sro in pairs if conf >= threshold]
                if reachable:
                    explored += 1
                scores.append(max(reachable) if reachable else 0.0)
            mean_score = float(np.mean(scores)) if scores else 0.0
            coverage = explored / len(per_query_pairs) if per_query_pairs else 0.0
            if verbose:
                print(
                    f"  min_conf={min_conf:.2f}  decay={decay:.2f}  "
                    f"score={mean_score:.4f}  coverage={coverage:.0%}"
                )
            candidates.append({
                "min_confidence": min_conf,
                "confidence_decay": decay,
                "threshold": threshold,
                "score": mean_score,
                "coverage": coverage,
            })

    if not candidates:
        return {}
    best_score = max(c["score"] for c in candidates)
    near_best = [c for c in candidates if c["score"] >= best_score - tolerance]
    best = max(near_best, key=lambda c: c["threshold"])
    return {k: v for k, v in best.items() if k != "threshold"}
