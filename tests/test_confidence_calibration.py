"""
Tests for confidence calibration utilities.

Runs in OFFLINE_MODE using the mock search API.
"""

import pytest
import numpy as np


class TestCalibrationResult:
    def test_empirical_precision_shape(self):
        from src.evaluation.confidence_calibration import _empirical_precision

        distances = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
        labels = np.array([1, 1, 0, 1, 0, 0])
        result = _empirical_precision(distances, labels, sigma=3.0, n_bins=5)

        assert result.sigma == 3.0
        assert 0.0 <= result.ece <= 1.0
        assert len(result.bin_accuracies) == 5
        assert len(result.bin_confidences) == 5
        assert len(result.bin_counts) == 5

    def test_perfect_calibration_has_low_ece(self):
        from src.evaluation.confidence_calibration import _empirical_precision

        # Construct perfectly calibrated data:
        # confidence = 0.9 → precision = 0.9, confidence = 0.1 → precision = 0.1
        n = 100
        distances = np.linspace(0, 5, n)
        sigma = 3.0
        confidences = np.exp(-distances / sigma)
        # Label each point with probability = its confidence
        rng = np.random.default_rng(0)
        labels = (rng.random(n) < confidences).astype(int)
        result = _empirical_precision(distances, labels, sigma=sigma, n_bins=10)
        # Perfect calibration not guaranteed at finite n, just check ECE is reasonable
        assert result.ece < 0.5

    def test_sigma_sweep_returns_correct_count(self):
        from src.evaluation.confidence_calibration import (
            SIGMA_GRID,
            _empirical_precision,
        )
        distances = np.abs(np.random.default_rng(1).standard_normal(50))
        labels = (np.random.default_rng(2).random(50) > 0.5).astype(int)
        results = [_empirical_precision(distances, labels, sigma=s) for s in SIGMA_GRID]
        assert len(results) == len(SIGMA_GRID)
        # Best sigma should have the lowest ECE
        best = min(results, key=lambda r: r.ece)
        assert best.sigma in SIGMA_GRID


class TestStatisticsUtils:
    def test_bootstrap_ci_contains_mean(self):
        from src.evaluation.statistics import bootstrap_ci

        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        mean, lo, hi = bootstrap_ci(scores, n_bootstrap=500)
        assert lo <= mean <= hi

    def test_bootstrap_ci_width(self):
        from src.evaluation.statistics import bootstrap_ci

        scores = list(range(100))
        mean, lo, hi = bootstrap_ci(scores, n_bootstrap=500)
        assert hi > lo  # CI has non-zero width

    def test_cohens_d_same_distributions(self):
        from src.evaluation.statistics import cohens_d

        a = [1.0, 1.0, 1.0, 1.0]
        d = cohens_d(a, a)
        assert abs(d) < 1e-6

    def test_cohens_d_different_means(self):
        from src.evaluation.statistics import cohens_d

        a = [0.0] * 20
        b = [1.0] * 20
        d = cohens_d(a, b)
        # Effect size should be large (|d| >> 0)
        assert abs(d) > 1.0

    def test_wilcoxon_table_symmetric(self):
        from src.evaluation.statistics import wilcoxon_table

        import numpy as np
        rng = np.random.default_rng(42)
        scores_a = list(rng.random(30))
        scores_b = list(rng.random(30))
        results = wilcoxon_table({"A": scores_a, "B": scores_b})
        # Should have one pair
        assert len(results) == 1
        key = ("A", "B")
        assert key in results
        assert "p_value" in results[key]
        assert 0.0 <= results[key]["p_value"] <= 1.0


class TestRecursiveGridSearch:
    def _make_search_api(self, confidences, formula="BaTiO3"):
        """Stub SearchAPI whose neighbors carry fixed confidences."""
        from src.schema import Neighbor

        class StubSearchAPI:
            def query_with_exclusion(self, comp, exclude_ids, n_neighbors):
                return [
                    Neighbor(
                        neighbor_index=i,
                        material_id=f"mp-{i}",
                        formula=formula,
                        distance=1.0,
                        confidence=c,
                    )
                    for i, c in enumerate(confidences[:n_neighbors])
                ]

        return StubSearchAPI()

    def test_picks_permissive_threshold_on_low_confidences(self, tiny_test_cases):
        """With realistic sigma=0.5 confidences (all < 0.2), only thresholds
        below 0.2 reach any neighbor — the repaired search must pick one.
        (BaTiO3 is in the fixture corpus, so reachable neighbors have SRO > 0.)"""
        from src.evaluation.confidence_calibration import (
            grid_search_recursive_params,
        )

        api = self._make_search_api([0.18, 0.15, 0.12])
        best = grid_search_recursive_params(
            tiny_test_cases, api, corpus=tiny_test_cases, verbose=False
        )
        assert best["min_confidence"] < 0.2
        assert best["confidence_decay"] < 0.2
        assert best["coverage"] > 0.0
        assert best["score"] > 0.0

    def test_score_uses_ground_truth_sro(self, tiny_test_cases):
        """Score must reflect precursor overlap, not raw confidence."""
        from src.evaluation.confidence_calibration import (
            grid_search_recursive_params,
        )

        # SrTiO3 is NOT in the fixture corpus; 0.9 clears every threshold
        api = self._make_search_api([0.9], formula="SrTiO3")
        best = grid_search_recursive_params(
            tiny_test_cases, api, corpus=tiny_test_cases, verbose=False
        )
        # SrTiO3 is not in the fixture corpus, so neighbor precursors are
        # empty and SRO with any non-empty query set is 0 — score must be 0,
        # not the 0.9 confidence.
        assert best["score"] == 0.0

    def test_recursive_defaults_explore_at_scale(self):
        """Regression: with sigma=0.5-scale confidences (0.15-0.35), the new
        defaults must let the search explore at least one child."""
        import inspect

        from src.recursive_synthesis import RecursiveSynthesisSearch

        sig = inspect.signature(RecursiveSynthesisSearch.__init__)
        min_confidence = sig.parameters["min_confidence"].default
        confidence_decay = sig.parameters["confidence_decay"].default

        # Depth-1 reachability rule from _recursive_search: a neighbor's
        # recipes are reachable iff conf >= 1.0*decay AND conf >= min_confidence
        realistic_confidences = [0.35, 0.25, 0.15]
        reachable = [
            c
            for c in realistic_confidences
            if c >= confidence_decay and c >= min_confidence
        ]
        assert reachable, (
            f"defaults (min_confidence={min_confidence}, "
            f"decay={confidence_decay}) prune every realistic neighbor"
        )
