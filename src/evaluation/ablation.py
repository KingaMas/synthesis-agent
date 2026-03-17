"""
Ablation studies for SKY.

A1 — MAGPIE vs MACE embeddings.
A2 — Recursive search vs direct KNN (coverage rate and recipe quality).
A3 — k sensitivity: retrieval quality vs k ∈ {1, 3, 5, 10, 20, 50}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.evaluation.benchmark import RetrievalBenchmark, BenchmarkResults
from src.evaluation.test_set_builder import TestCase, build_test_set


# ---------------------------------------------------------------------------
# A1: MAGPIE vs MACE
# ---------------------------------------------------------------------------

def run_embedding_ablation(
    test_cases: Optional[list[TestCase]] = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    max_cases: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, BenchmarkResults]:
    """Compare MAGPIE and MACE retrievers on the same test set.

    Requires the HDF5 embedding files to be present.

    Returns:
        Dict with keys 'MAGPIE' and 'MACE', each a BenchmarkResults object.
    """
    from src.search_api import SearchAPI
    from src.embedding import InputType
    from pymatgen.core import Composition

    if test_cases is None:
        test_cases = build_test_set()
    if max_cases is not None:
        test_cases = test_cases[:max_cases]

    benchmark = RetrievalBenchmark(
        test_cases=test_cases,
        k_values=list(k_values),
        verbose=verbose,
    )

    results: dict[str, BenchmarkResults] = {}

    for name, input_type in [("MAGPIE", InputType.COMPOSITION), ("MACE", InputType.STRUCTURE)]:
        if verbose:
            print(f"\n--- {name} embedding ablation ---")
        try:
            api = SearchAPI(input_type=input_type, max_neighbors=max(k_values) + 10)
        except Exception as e:
            if verbose:
                print(f"  Skipping {name}: {e}")
            continue

        class _Retriever:
            def __init__(self, search_api, _test_cases, _input_type):
                self._api = search_api
                self._lookup = {tc.reduced_formula: tc for tc in _test_cases}
                self._input_type = _input_type

            def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
                try:
                    comp = Composition(query.reduced_formula)
                    neighbors = self._api.query_with_exclusion(
                        comp,
                        exclude_ids=[query.material_id] if query.material_id else [],
                        n_neighbors=k,
                    )
                    tcs = []
                    for n in neighbors:
                        tc = self._lookup.get(n.formula)
                        if tc and tc.reduced_formula != query.reduced_formula:
                            tcs.append(tc)
                    return tcs[:k]
                except Exception:
                    return []

        retriever = _Retriever(api, test_cases, input_type)
        results[name] = benchmark.evaluate(retriever, retriever_name=name)

    return results


# ---------------------------------------------------------------------------
# A2: Recursive search benefit
# ---------------------------------------------------------------------------

@dataclass
class RecursiveBenefitResult:
    """Results of the recursive vs direct KNN ablation."""
    n_test: int
    direct_coverage: float      # fraction of materials with >= 1 recipe
    recursive_coverage: float
    direct_mean_sro: float      # mean SRO@5 among covered materials
    recursive_mean_sro: float


def run_recursive_ablation(
    test_cases: Optional[list[TestCase]] = None,
    max_cases: int = 100,
    verbose: bool = True,
) -> RecursiveBenefitResult:
    """Compare recursive search vs direct KNN on rare-material coverage.

    Uses test cases that have no direct recipe match among their top-5 neighbors
    as a proxy for "rare" materials.

    Note: Full recursive search is expensive (makes MP API calls).
    Set max_cases to a small number for a quick ablation.
    """
    from src.search_api import SearchAPI
    from src.embedding import InputType
    from src.evaluation.benchmark import sro_at_k
    from pymatgen.core import Composition

    if test_cases is None:
        test_cases = build_test_set()
    test_cases = test_cases[:max_cases]

    api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=50)
    lookup = {tc.reduced_formula: tc for tc in test_cases}

    direct_covered = 0
    recursive_covered = 0
    direct_sros: list[float] = []
    recursive_sros: list[float] = []

    for i, tc in enumerate(test_cases):
        if verbose and i % 20 == 0:
            print(f"  Recursive ablation: {i}/{len(test_cases)}")
        try:
            comp = Composition(tc.reduced_formula)
            neighbors = api.query_with_exclusion(
                comp,
                exclude_ids=[tc.material_id] if tc.material_id else [],
                n_neighbors=5,
            )
            neighbor_tcs = [
                lookup[n.formula] for n in neighbors if n.formula in lookup
            ]
        except Exception:
            continue

        # Direct coverage: at least one top-5 neighbor shares >= 1 precursor element
        q_prec = set(tc.precursor_elements)
        direct_hit = any(
            len(q_prec & set(n.precursor_elements)) > 0 for n in neighbor_tcs
        )
        if direct_hit:
            direct_covered += 1
            direct_sros.append(sro_at_k(tc, neighbor_tcs, k=5))

        # Recursive coverage proxy: expand to depth-2 by also fetching
        # neighbors of each neighbor (first neighbor only for efficiency)
        recursive_hit = direct_hit
        if not direct_hit and neighbor_tcs:
            try:
                first_neighbor_comp = Composition(neighbor_tcs[0].reduced_formula)
                depth2_neighbors = api.query_with_exclusion(
                    first_neighbor_comp,
                    exclude_ids=[tc.material_id] if tc.material_id else [],
                    n_neighbors=5,
                )
                d2_tcs = [
                    lookup[n.formula] for n in depth2_neighbors
                    if n.formula in lookup and n.formula != tc.reduced_formula
                ]
                recursive_hit = any(
                    len(q_prec & set(n.precursor_elements)) > 0 for n in d2_tcs
                )
                if recursive_hit:
                    recursive_sros.append(sro_at_k(tc, d2_tcs, k=5))
            except Exception:
                pass

        if recursive_hit:
            recursive_covered += 1

    n = len(test_cases)
    return RecursiveBenefitResult(
        n_test=n,
        direct_coverage=direct_covered / n if n > 0 else 0.0,
        recursive_coverage=recursive_covered / n if n > 0 else 0.0,
        direct_mean_sro=float(np.mean(direct_sros)) if direct_sros else 0.0,
        recursive_mean_sro=float(np.mean(recursive_sros + direct_sros)) if (recursive_sros or direct_sros) else 0.0,
    )


# ---------------------------------------------------------------------------
# A3: k sensitivity
# ---------------------------------------------------------------------------

def run_k_sensitivity(
    test_cases: Optional[list[TestCase]] = None,
    k_grid: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
    max_cases: Optional[int] = None,
    verbose: bool = True,
) -> BenchmarkResults:
    """Plot retrieval quality (SRO, MCR, NDCG) vs k on the MAGPIE retriever."""
    from src.search_api import SearchAPI
    from src.embedding import InputType
    from pymatgen.core import Composition

    if test_cases is None:
        test_cases = build_test_set()
    if max_cases is not None:
        test_cases = test_cases[:max_cases]

    api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=max(k_grid) + 10)
    lookup = {tc.reduced_formula: tc for tc in test_cases}

    benchmark = RetrievalBenchmark(
        test_cases=test_cases,
        k_values=list(k_grid),
        verbose=verbose,
    )

    class _MAGPIERetriever:
        def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
            try:
                comp = Composition(query.reduced_formula)
                neighbors = api.query_with_exclusion(
                    comp,
                    exclude_ids=[query.material_id] if query.material_id else [],
                    n_neighbors=k,
                )
                return [
                    lookup[n.formula] for n in neighbors
                    if n.formula in lookup and n.formula != query.reduced_formula
                ][:k]
            except Exception:
                return []

    return benchmark.evaluate(_MAGPIERetriever(), retriever_name="MAGPIE_k_sensitivity")
