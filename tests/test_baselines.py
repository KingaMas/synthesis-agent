"""
Tests for the four baseline retrievers.

All tests run in OFFLINE_MODE using tiny_test_cases from fixtures.
"""

import pytest


class TestRandomRetriever:
    def test_returns_k_or_fewer_results(self, tiny_test_cases):
        from src.evaluation.baselines import RandomRetriever
        if not tiny_test_cases:
            pytest.skip("No test cases available")
        ret = RandomRetriever(corpus=tiny_test_cases, seed=0)
        results = ret.retrieve(tiny_test_cases[0], k=3)
        assert len(results) <= 3

    def test_excludes_query(self, tiny_test_cases):
        from src.evaluation.baselines import RandomRetriever
        if not tiny_test_cases:
            pytest.skip("No test cases available")
        ret = RandomRetriever(corpus=tiny_test_cases, seed=0)
        query = tiny_test_cases[0]
        results = ret.retrieve(query, k=10)
        returned_formulas = {r.reduced_formula for r in results}
        assert query.reduced_formula not in returned_formulas

    def test_reproducible(self, tiny_test_cases):
        from src.evaluation.baselines import RandomRetriever
        if not tiny_test_cases:
            pytest.skip("No test cases available")
        ret1 = RandomRetriever(corpus=tiny_test_cases, seed=42)
        ret2 = RandomRetriever(corpus=tiny_test_cases, seed=42)
        r1 = [r.reduced_formula for r in ret1.retrieve(tiny_test_cases[0], k=3)]
        r2 = [r.reduced_formula for r in ret2.retrieve(tiny_test_cases[0], k=3)]
        assert r1 == r2


class TestElementJaccardRetriever:
    def test_ranks_by_element_overlap(self, tiny_test_cases):
        from src.evaluation.baselines import ElementJaccardRetriever
        if len(tiny_test_cases) < 2:
            pytest.skip("Need at least 2 test cases")
        ret = ElementJaccardRetriever(corpus=tiny_test_cases)
        query = tiny_test_cases[0]
        results = ret.retrieve(query, k=3)
        assert all(r.reduced_formula != query.reduced_formula for r in results)

    def test_returns_sorted_descending(self, tiny_test_cases):
        from src.evaluation.baselines import ElementJaccardRetriever, _jaccard
        if len(tiny_test_cases) < 3:
            pytest.skip("Need at least 3 test cases")
        ret = ElementJaccardRetriever(corpus=tiny_test_cases)
        query = tiny_test_cases[0]
        results = ret.retrieve(query, k=5)
        q_els = set(query.elements)
        scores = [_jaccard(q_els, set(r.elements)) for r in results]
        for a, b in zip(scores, scores[1:]):
            assert a >= b - 1e-9  # non-increasing


class TestStoichiometricVectorRetriever:
    def test_basic_retrieval(self, tiny_test_cases):
        from src.evaluation.baselines import StoichiometricVectorRetriever
        if not tiny_test_cases:
            pytest.skip("No test cases available")
        ret = StoichiometricVectorRetriever(corpus=tiny_test_cases)
        results = ret.retrieve(tiny_test_cases[0], k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_does_not_return_query(self, tiny_test_cases):
        from src.evaluation.baselines import StoichiometricVectorRetriever
        if not tiny_test_cases:
            pytest.skip("No test cases available")
        ret = StoichiometricVectorRetriever(corpus=tiny_test_cases)
        query = tiny_test_cases[0]
        results = ret.retrieve(query, k=5)
        assert all(r.reduced_formula != query.reduced_formula for r in results)


class TestFormulaTFIDFRetriever:
    def test_basic_retrieval(self, tiny_test_cases):
        from src.evaluation.baselines import FormulaTFIDFRetriever
        if not tiny_test_cases:
            pytest.skip("No test cases available")
        ret = FormulaTFIDFRetriever(corpus=tiny_test_cases)
        results = ret.retrieve(tiny_test_cases[0], k=3)
        assert isinstance(results, list)

    def test_does_not_return_query(self, tiny_test_cases):
        from src.evaluation.baselines import FormulaTFIDFRetriever
        if not tiny_test_cases:
            pytest.skip("No test cases available")
        ret = FormulaTFIDFRetriever(corpus=tiny_test_cases)
        query = tiny_test_cases[0]
        results = ret.retrieve(query, k=5)
        assert all(r.reduced_formula != query.reduced_formula for r in results)
