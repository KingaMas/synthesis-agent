"""
Retrieval benchmark for SKY synthesis agent.

Constructs leave-one-out evaluation over mp_synthesis_recipes: holds each
material out from the KNN index and measures whether its top-k neighbors
share synthesis characteristics.

Metrics
-------
SRO@k   Synthesis Recipe Overlap — mean Jaccard similarity of precursor
        element sets between held-out material and its k retrieved neighbors.
        This is the primary metric reported in Table 1.
MCR@k   Method Consistency Rate — fraction of top-k neighbors sharing the
        same synthesis method family as the query.
NDCG@k  Standard IR metric with SRO as graded relevance.
MRR     Mean Reciprocal Rank — position of first neighbor with SRO >= 0.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np
from pymatgen.core import Composition

from src.evaluation.test_set_builder import TestCase, build_test_set


# ---------------------------------------------------------------------------
# Retriever protocol – implemented by SearchAPI wrappers and all baselines
# ---------------------------------------------------------------------------

class BaselineRetriever(Protocol):
    """Common interface for all retrievers (MAGPIE, MACE, baselines)."""

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        """Return up to k neighbours for *query*, excluding query itself."""
        ...


# ---------------------------------------------------------------------------
# Per-query metric helpers
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def sro_at_k(query: TestCase, neighbors: list[TestCase], k: int) -> float:
    """Mean Jaccard similarity of precursor element sets (top-k)."""
    query_prec = set(query.precursor_elements)
    scores = [
        _jaccard(query_prec, set(n.precursor_elements))
        for n in neighbors[:k]
    ]
    return float(np.mean(scores)) if scores else 0.0


def mcr_at_k(query: TestCase, neighbors: list[TestCase], k: int) -> float:
    """Fraction of top-k neighbors sharing same synthesis method family."""
    if not neighbors:
        return 0.0
    hits = sum(
        1 for n in neighbors[:k] if n.synthesis_method == query.synthesis_method
    )
    return hits / min(k, len(neighbors))


def ndcg_at_k(query: TestCase, neighbors: list[TestCase], k: int) -> float:
    """NDCG@k using SRO as graded relevance."""
    query_prec = set(query.precursor_elements)

    def gain(n: TestCase) -> float:
        return _jaccard(query_prec, set(n.precursor_elements))

    dcg = sum(
        gain(n) / math.log2(i + 2)
        for i, n in enumerate(neighbors[:k])
    )
    # Ideal DCG: sort by relevance descending
    ideal_gains = sorted(
        [gain(n) for n in neighbors[:k]], reverse=True
    )
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(query: TestCase, neighbors: list[TestCase], threshold: float = 0.3) -> float:
    """Reciprocal rank of first neighbor with SRO >= threshold."""
    query_prec = set(query.precursor_elements)
    for rank, n in enumerate(neighbors, start=1):
        if _jaccard(query_prec, set(n.precursor_elements)) >= threshold:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResults:
    """Aggregate benchmark results for one retriever."""

    retriever_name: str
    k_values: list[int]
    n_queries: int

    # Per-k aggregates (mean over all queries)
    sro: dict[int, float] = field(default_factory=dict)
    mcr: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)

    # MRR is k-independent
    mean_mrr: float = 0.0

    # Per-query raw scores for bootstrap CIs
    per_query_sro: dict[int, list[float]] = field(default_factory=dict)
    per_query_mcr: dict[int, list[float]] = field(default_factory=dict)
    per_query_ndcg: dict[int, list[float]] = field(default_factory=dict)
    per_query_mrr: list[float] = field(default_factory=list)

    def summary_table(self) -> str:
        """Return a compact ASCII table for quick inspection."""
        lines = [
            f"Retriever: {self.retriever_name}  (n={self.n_queries})",
            f"{'k':>4}  {'SRO':>8}  {'MCR':>8}  {'NDCG':>8}",
            "-" * 36,
        ]
        for k in self.k_values:
            lines.append(
                f"{k:>4}  {self.sro.get(k, float('nan')):>8.4f}  "
                f"{self.mcr.get(k, float('nan')):>8.4f}  "
                f"{self.ndcg.get(k, float('nan')):>8.4f}"
            )
        lines.append(f"MRR (threshold=0.3): {self.mean_mrr:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class RetrievalBenchmark:
    """Run leave-one-out retrieval evaluation over a test set."""

    def __init__(
        self,
        test_cases: Optional[list[TestCase]] = None,
        k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
        max_cases: Optional[int] = None,
        verbose: bool = True,
    ):
        if test_cases is None:
            test_cases = build_test_set()
        if max_cases is not None:
            test_cases = test_cases[:max_cases]
        self.test_cases = test_cases
        self.k_values = list(k_values)
        self.verbose = verbose
        self._max_k = max(k_values)

    def evaluate(
        self,
        retriever: BaselineRetriever,
        retriever_name: str = "unnamed",
    ) -> BenchmarkResults:
        """Evaluate *retriever* over the full test set.

        Args:
            retriever: Any object implementing ``retrieve(query, k)``.
            retriever_name: Label used in result tables and figures.

        Returns:
            BenchmarkResults with per-k and per-query scores.
        """
        results = BenchmarkResults(
            retriever_name=retriever_name,
            k_values=self.k_values,
            n_queries=len(self.test_cases),
        )
        # Initialise per-query lists
        for k in self.k_values:
            results.per_query_sro[k] = []
            results.per_query_mcr[k] = []
            results.per_query_ndcg[k] = []

        for i, query in enumerate(self.test_cases):
            if self.verbose and i % 100 == 0:
                print(f"  [{retriever_name}] {i}/{len(self.test_cases)}")

            neighbors = retriever.retrieve(query, k=self._max_k)

            for k in self.k_values:
                results.per_query_sro[k].append(sro_at_k(query, neighbors, k))
                results.per_query_mcr[k].append(mcr_at_k(query, neighbors, k))
                results.per_query_ndcg[k].append(ndcg_at_k(query, neighbors, k))

            results.per_query_mrr.append(mrr(query, neighbors))

        # Aggregate
        for k in self.k_values:
            results.sro[k] = float(np.mean(results.per_query_sro[k]))
            results.mcr[k] = float(np.mean(results.per_query_mcr[k]))
            results.ndcg[k] = float(np.mean(results.per_query_ndcg[k]))
        results.mean_mrr = float(np.mean(results.per_query_mrr))

        if self.verbose:
            print(results.summary_table())

        return results
