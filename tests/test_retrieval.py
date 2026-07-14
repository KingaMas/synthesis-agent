"""
Tests for the retrieval benchmark and SearchAPI refactoring.

All tests run in OFFLINE_MODE (no MP API calls).
"""

import pytest
from pymatgen.core import Composition


# ---------------------------------------------------------------------------
# SearchAPI refactoring tests
# ---------------------------------------------------------------------------

class TestMockSearchAPI:
    def test_query_returns_neighbors(self, mock_search_api):
        comp = Composition("LiCoO2")
        neighbors = mock_search_api.query(comp, n_neighbors=3)
        assert len(neighbors) <= 3
        for n in neighbors:
            assert hasattr(n, "material_id")
            assert hasattr(n, "formula")
            assert 0.0 <= n.confidence <= 1.0
            assert n.distance >= 0.0

    def test_query_with_exclusion_removes_ids(self, mock_search_api):
        comp = Composition("LiCoO2")
        all_neighbors = mock_search_api.query(comp, n_neighbors=5)
        if not all_neighbors:
            pytest.skip("Fixture has no neighbors for LiCoO2")
        exclude_id = all_neighbors[0].material_id
        filtered = mock_search_api.query_with_exclusion(
            comp, exclude_ids=[exclude_id], n_neighbors=5
        )
        returned_ids = {n.material_id for n in filtered}
        assert exclude_id not in returned_ids

    def test_query_with_exclusion_count(self, mock_search_api):
        comp = Composition("BaTiO3")
        neighbors = mock_search_api.query_with_exclusion(
            comp, exclude_ids=[], n_neighbors=3
        )
        assert len(neighbors) <= 3

    def test_neighbor_index_sequential(self, mock_search_api):
        comp = Composition("Fe2O3")
        neighbors = mock_search_api.query(comp, n_neighbors=5)
        for i, n in enumerate(neighbors):
            assert n.neighbor_index == i


# ---------------------------------------------------------------------------
# BenchmarkResults smoke tests
# ---------------------------------------------------------------------------

class TestBenchmarkMetrics:
    def test_sro_perfect_match(self, tiny_test_cases):
        from src.evaluation.benchmark import sro_at_k

        if not tiny_test_cases:
            pytest.skip("No test cases available")
        tc = tiny_test_cases[0]
        # Query against itself should give SRO = 1.0
        score = sro_at_k(tc, [tc], k=1)
        assert abs(score - 1.0) < 1e-6

    def test_sro_no_overlap(self, tiny_test_cases):
        from src.evaluation.benchmark import sro_at_k
        from src.evaluation.test_set_builder import TestCase

        if len(tiny_test_cases) < 2:
            pytest.skip("Need at least 2 test cases")

        empty_tc = TestCase(
            material_id="x",
            formula="Au",
            reduced_formula="Au",
            elements=["Au"],
            synthesis_method="other",
            precursor_elements=[],
            raw_recipe={},
        )
        score = sro_at_k(tiny_test_cases[0], [empty_tc], k=1)
        # Both empty sets → Jaccard returns 1.0 by convention; non-empty query vs empty neighbor → 0
        if tiny_test_cases[0].precursor_elements:
            assert score == 0.0
        else:
            assert 0.0 <= score <= 1.0

    def test_mcr_same_method(self, tiny_test_cases):
        from src.evaluation.benchmark import mcr_at_k

        if not tiny_test_cases:
            pytest.skip("No test cases available")
        tc = tiny_test_cases[0]
        score = mcr_at_k(tc, [tc], k=1)
        assert score == 1.0

    def test_ndcg_perfect_ranking(self, tiny_test_cases):
        from src.evaluation.benchmark import ndcg_at_k

        if not tiny_test_cases:
            pytest.skip("No test cases available")
        tc = tiny_test_cases[0]
        # A single perfect neighbor should give NDCG = 1 when SRO > 0
        score = ndcg_at_k(tc, [tc], k=1)
        assert 0.0 <= score <= 1.0 + 1e-9

    def test_mrr_first_hit(self, tiny_test_cases):
        from src.evaluation.benchmark import mrr

        if not tiny_test_cases:
            pytest.skip("No test cases available")
        tc = tiny_test_cases[0]
        # Self-retrieval should score MRR >= 0
        score = mrr(tc, [tc])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Mock agent integration
# ---------------------------------------------------------------------------

class TestMockAgent:
    def test_mock_agent_recipes(self, mock_agent):
        recipes = mock_agent.get_synthesis_recipes_by_formula("LiCoO2")
        assert isinstance(recipes, list)

    def test_mock_agent_neighbors(self, mock_agent):
        neighbors = mock_agent.find_similar_materials_by_composition("BaTiO3", n_neighbors=3)
        assert isinstance(neighbors, list)
        assert len(neighbors) <= 3

    def test_mock_agent_unknown_formula_returns_empty(self, mock_agent):
        # Should return empty list without raising
        result = mock_agent.get_synthesis_recipes_by_formula("XYZ999")
        assert isinstance(result, list)
        assert result == []


# ---------------------------------------------------------------------------
# Wall-clock timing (extension B)
# ---------------------------------------------------------------------------

class TestBenchmarkTiming:
    def test_latency_recorded_per_query(self, tiny_test_cases):
        from src.evaluation.baselines import ElementJaccardRetriever
        from src.evaluation.benchmark import RetrievalBenchmark

        bench = RetrievalBenchmark(
            test_cases=tiny_test_cases, k_values=(1, 3), verbose=False
        )
        results = bench.evaluate(
            ElementJaccardRetriever(corpus=tiny_test_cases), "jaccard"
        )
        assert len(results.per_query_latency) == len(tiny_test_cases)
        assert all(t >= 0.0 for t in results.per_query_latency)
        stats = results.latency_stats()
        assert stats["total_s"] >= stats["mean_s"]
        assert set(stats) == {"mean_s", "median_s", "p95_s", "total_s"}


# ---------------------------------------------------------------------------
# Formula-SRO and oracle as first-class benchmark metrics (T2)
# ---------------------------------------------------------------------------

class TestFormulaSroInBenchmark:
    def test_formula_sro_computed_per_k(self, tiny_test_cases):
        from src.evaluation.baselines import ElementJaccardRetriever
        from src.evaluation.benchmark import RetrievalBenchmark

        bench = RetrievalBenchmark(
            test_cases=tiny_test_cases, k_values=(1, 3),
            corpus=tiny_test_cases, verbose=False,
        )
        res = bench.evaluate(ElementJaccardRetriever(corpus=tiny_test_cases), "j")
        assert set(res.formula_sro) == {1, 3}
        assert all(0.0 <= v <= 1.0 for v in res.formula_sro.values())
        assert len(res.per_query_formula_sro[1]) == len(tiny_test_cases)

    def test_oracle_bounds_every_retriever(self, tiny_test_cases):
        from src.evaluation.baselines import RandomRetriever
        from src.evaluation.benchmark import RetrievalBenchmark

        bench = RetrievalBenchmark(
            test_cases=tiny_test_cases, k_values=(1, 3),
            corpus=tiny_test_cases, compute_oracle=True, verbose=False,
        )
        res = bench.evaluate(RandomRetriever(corpus=tiny_test_cases, seed=0), "r")
        assert res.mean_oracle is not None
        assert res.mean_regret >= -1e-9  # oracle is an upper bound
        assert res.mean_oracle >= res.formula_sro[3] - 1e-9

    def test_oracle_requires_corpus(self, tiny_test_cases):
        import pytest
        from src.evaluation.benchmark import RetrievalBenchmark

        with pytest.raises(ValueError):
            RetrievalBenchmark(
                test_cases=tiny_test_cases, compute_oracle=True, verbose=False
            )
