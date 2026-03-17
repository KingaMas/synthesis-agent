"""
Statistical analysis utilities for evaluation results.

All tables in the paper report mean ± 95% CI with significance markers.

Functions
---------
bootstrap_ci          — Bootstrap 95% CIs for a list of per-query scores.
wilcoxon_table        — Pairwise Wilcoxon signed-rank tests for all methods.
cohens_d              — Effect size between two score distributions.
format_results_table  — LaTeX-ready comparison table.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        scores:      Per-query metric values.
        n_bootstrap: Number of bootstrap resamples.
        ci:          Confidence level (default 0.95).
        seed:        Random seed.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=np.float64)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_table(
    method_scores: dict[str, list[float]],
    alpha: float = 0.05,
) -> dict[tuple[str, str], dict]:
    """Pairwise Wilcoxon signed-rank tests for all method pairs.

    Args:
        method_scores: Dict mapping method name → per-query scores list.
        alpha:         Significance level.

    Returns:
        Dict keyed by (method_a, method_b) with keys:
          statistic, p_value, significant, effect_size (Cohen's d).
    """
    methods = list(method_scores.keys())
    results: dict[tuple[str, str], dict] = {}

    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            scores_a = np.array(method_scores[a], dtype=np.float64)
            scores_b = np.array(method_scores[b], dtype=np.float64)

            # Wilcoxon requires equal-length paired samples
            n = min(len(scores_a), len(scores_b))
            scores_a = scores_a[:n]
            scores_b = scores_b[:n]

            try:
                stat, p = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox")
            except ValueError:
                # All differences are zero
                stat, p = 0.0, 1.0

            d = cohens_d(list(scores_a), list(scores_b))
            results[(a, b)] = {
                "statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < alpha),
                "effect_size": d,
            }

    return results


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------

def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size between two distributions.

    Uses pooled standard deviation.
    """
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    n_a, n_b = len(arr_a), len(arr_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((n_a - 1) * arr_a.std(ddof=1) ** 2 + (n_b - 1) * arr_b.std(ddof=1) ** 2)
        / (n_a + n_b - 2)
    )
    mean_diff = arr_a.mean() - arr_b.mean()
    if pooled_std < 1e-12:
        # Zero variance in both groups: effect is infinite if means differ, zero otherwise
        return float("inf") if abs(mean_diff) > 1e-12 else 0.0
    return float(mean_diff / pooled_std)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_results_table(
    method_results: dict,
    metric: str = "SRO@5",
    k: int = 5,
    alpha: float = 0.05,
) -> str:
    """Build a LaTeX-ready comparison table with CIs and significance stars.

    Args:
        method_results: Dict mapping method name → BenchmarkResults object.
        metric:         Display metric name.
        k:              Which k-value to report.
        alpha:          Significance threshold for stars.

    Returns:
        LaTeX tabular string.
    """
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Method & {metric} & 95\% CI & Sig. \\",
        r"\midrule",
    ]

    all_scores: dict[str, list[float]] = {}
    for name, res in method_results.items():
        all_scores[name] = res.per_query_sro.get(k, [])

    # Wilcoxon vs best method
    best_name = max(all_scores, key=lambda n: np.mean(all_scores[n]) if all_scores[n] else -1)
    pw = wilcoxon_table(all_scores)

    for name, res in method_results.items():
        scores = all_scores.get(name, [])
        if not scores:
            lines.append(rf"{name} & -- & -- & -- \\")
            continue

        mean, lo, hi = bootstrap_ci(scores)
        key = (best_name, name) if (best_name, name) in pw else (name, best_name)
        sig_info = pw.get(key, {})
        sig_marker = "*" if sig_info.get("significant") and name != best_name else ""

        lines.append(
            rf"{name} & {mean:.4f}{sig_marker} & [{lo:.4f}, {hi:.4f}] & "
            rf"d={sig_info.get('effect_size', float('nan')):.2f} \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
