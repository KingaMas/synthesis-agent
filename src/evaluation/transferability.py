"""
Recipe-transferability metrics for retrieval evaluation.

The element-level SRO in benchmark.py is partly circular: element-overlap
retrievers are scored by element overlap. This module scores retrieval by
FORMULA-level recipe transfer — Jaccard similarity over normalized
precursor reduced-formula sets — and calibrates every retriever against
the oracle ceiling (the best formula-SRO any retriever could achieve on
the same corpus).

Metrics
-------
formula_sro_at_k    Mean Jaccard of precursor formula sets, top-k.
oracle ceiling      Mean over the k highest-achievable Jaccards (leave-
                    one-out over the corpus).
regret              oracle - achieved, per query.
Route stratification: all of the above per synthesis method family.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.evaluation.benchmark import BaselineRetriever, _jaccard
from src.evaluation.planning import normalize_formula_set, _extract_precursor_formulas
from src.evaluation.statistics import bootstrap_ci
from src.evaluation.test_set_builder import TestCase


class FormulaSetIndex:
    """Cache of normalized precursor formula sets per corpus material."""

    def __init__(self, corpus: list[TestCase]):
        self._sets: dict[str, frozenset[str]] = {}
        for tc in corpus:
            self._sets[tc.reduced_formula] = frozenset(
                normalize_formula_set(_extract_precursor_formulas(tc.raw_recipe))
            )

    def get(self, tc: TestCase) -> frozenset[str]:
        s = self._sets.get(tc.reduced_formula)
        if s is None:
            s = frozenset(
                normalize_formula_set(_extract_precursor_formulas(tc.raw_recipe))
            )
            self._sets[tc.reduced_formula] = s
        return s


def formula_sro_at_k(
    query: TestCase,
    neighbors: list[TestCase],
    k: int,
    index: FormulaSetIndex,
) -> float:
    """Mean Jaccard of normalized precursor formula sets (top-k)."""
    q = index.get(query)
    scores = [_jaccard(set(q), set(index.get(n))) for n in neighbors[:k]]
    return float(np.mean(scores)) if scores else 0.0


def oracle_formula_sro_at_k(
    query: TestCase,
    corpus: list[TestCase],
    k: int,
    index: FormulaSetIndex,
) -> float:
    """Ceiling: mean of the k highest formula-set Jaccards over the corpus.

    Leave-one-out: the query's own formula is excluded. This is the best
    any retriever could score on formula-SRO@k for this query.
    """
    q = set(index.get(query))
    scores = sorted(
        (
            _jaccard(q, set(index.get(tc)))
            for tc in corpus
            if tc.reduced_formula != query.reduced_formula
        ),
        reverse=True,
    )
    top = scores[:k]
    return float(np.mean(top)) if top else 0.0


def evaluate_transferability(
    retrievers: dict[str, BaselineRetriever],
    test_cases: list[TestCase],
    corpus: list[TestCase],
    k: int = 5,
    verbose: bool = True,
) -> dict:
    """Score retrievers on formula-SRO@k against the oracle ceiling.

    Returns a dict with, per retriever and for the oracle: mean [95% CI],
    per-route means, mean regret (oracle - achieved), and the query list
    (so the exact evaluation sample is stored with the results).
    """
    index = FormulaSetIndex(corpus)

    if verbose:
        print(f"Computing oracle ceiling for {len(test_cases)} queries ...")
    oracle_scores = []
    for i, q in enumerate(test_cases):
        if verbose and i % 100 == 0:
            print(f"  oracle {i}/{len(test_cases)}")
        oracle_scores.append(oracle_formula_sro_at_k(q, corpus, k, index))

    def _aggregate(scores: list[float]) -> dict:
        mean, lo, hi = bootstrap_ci(scores)
        per_route: dict[str, dict] = {}
        for route in sorted({q.synthesis_method for q in test_cases}):
            route_scores = [
                s for s, q in zip(scores, test_cases) if q.synthesis_method == route
            ]
            if route_scores:
                r_mean, r_lo, r_hi = bootstrap_ci(route_scores)
                per_route[route] = {
                    "mean": r_mean, "ci_lo": r_lo, "ci_hi": r_hi,
                    "n": len(route_scores),
                }
        return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "per_route": per_route}

    results: dict = {
        "_config": {
            "k": k,
            "n_queries": len(test_cases),
            "corpus_size": len(corpus),
            "query_formulas": [q.reduced_formula for q in test_cases],
        },
        "oracle": {**_aggregate(oracle_scores), "per_query": oracle_scores},
    }

    for name, retriever in retrievers.items():
        if verbose:
            print(f"Evaluating {name} ...")
        scores = []
        for q in test_cases:
            neighbors = retriever.retrieve(q, k=k)
            scores.append(formula_sro_at_k(q, neighbors, k, index))
        regret = [o - s for o, s in zip(oracle_scores, scores)]
        results[name] = {
            **_aggregate(scores),
            "mean_regret": float(np.mean(regret)),
            "frac_regret_gt_0.2": float(np.mean([r > 0.2 for r in regret])),
            "per_query": scores,
        }
        if verbose:
            print(f"  {name}: formula-SRO@{k} = {results[name]['mean']:.3f} "
                  f"(oracle {results['oracle']['mean']:.3f})")

    return results
