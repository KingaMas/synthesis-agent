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
