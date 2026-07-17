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


class TestHybridRRFRetriever:
    def test_excludes_query_and_dedupes(self, tiny_test_cases):
        from src.evaluation.baselines import (
            ElementJaccardRetriever,
            HybridRRFRetriever,
            StoichiometricVectorRetriever,
        )
        if len(tiny_test_cases) < 3:
            pytest.skip("Need at least 3 test cases")
        hybrid = HybridRRFRetriever([
            ElementJaccardRetriever(corpus=tiny_test_cases),
            StoichiometricVectorRetriever(corpus=tiny_test_cases),
        ])
        query = tiny_test_cases[0]
        results = hybrid.retrieve(query, k=5)
        formulas = [r.reduced_formula for r in results]
        assert query.reduced_formula not in formulas
        assert len(formulas) == len(set(formulas))

    def test_consensus_ranks_above_single_source(self, tiny_test_cases):
        from src.evaluation.baselines import HybridRRFRetriever

        if len(tiny_test_cases) < 3:
            pytest.skip("Need at least 3 test cases")
        a, b, c = tiny_test_cases[0], tiny_test_cases[1], tiny_test_cases[2]

        class FixedRetriever:
            def __init__(self, ranking):
                self.ranking = ranking

            def retrieve(self, query, k):
                return self.ranking[:k]

        # b is found by both components; c by only one
        hybrid = HybridRRFRetriever([FixedRetriever([b, c]), FixedRetriever([b])])
        results = hybrid.retrieve(a, k=2)
        assert results[0].reduced_formula == b.reduced_formula


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


class TestFormRerank:
    def test_precursor_form_classes(self):
        from src.evaluation.form_rerank import precursor_form
        assert precursor_form("BaCO3") == "carbonate"
        assert precursor_form("Fe(NO3)3") == "nitrate"
        assert precursor_form("TiO2") == "oxide"
        assert precursor_form("Ba(OH)2") == "hydroxide"
        assert precursor_form("CH3COOLi") == "organic"
        assert precursor_form("FeCl3") == "halide"
        assert precursor_form("Fe") == "element"

    def test_reranker_implements_protocol(self, tiny_test_cases):
        from src.evaluation.baselines import ElementJaccardRetriever
        from src.evaluation.form_rerank import (
            FormCooccurrenceModel,
            FormRerankRetriever,
        )
        base = ElementJaccardRetriever(corpus=tiny_test_cases)
        model = FormCooccurrenceModel(tiny_test_cases)
        rr = FormRerankRetriever(base, model, mode="max_route")
        query = tiny_test_cases[0]
        results = rr.retrieve(query, k=3)
        assert len(results) <= 3
        assert all(r.reduced_formula != query.reduced_formula for r in results)


class TestPairwiseRank:
    def test_feature_vector_shape_and_finiteness(self, tiny_test_cases):
        from src.evaluation.pairwise_rank import PairFeatureExtractor

        ex = PairFeatureExtractor(tiny_test_cases)
        f = ex.features(tiny_test_cases[0], tiny_test_cases[1])
        assert len(f) == len(ex.feature_names)
        assert all(x == x for x in f)  # no NaNs

    def test_ranker_implements_protocol(self, tiny_test_cases):
        from src.evaluation.baselines import ElementJaccardRetriever
        from src.evaluation.pairwise_rank import (
            GradientBoostedRanker,
            PairFeatureExtractor,
            PairwiseRankRetriever,
        )

        ex = PairFeatureExtractor(tiny_test_cases)
        base = ElementJaccardRetriever(corpus=tiny_test_cases)
        ranker = GradientBoostedRanker(
            ex, n_candidates=5, n_estimators=5, min_child_samples=1
        ).fit(tiny_test_cases, base, verbose=False)
        query = tiny_test_cases[0]
        results = PairwiseRankRetriever(base, ranker, fetch_factor=2).retrieve(
            query, k=3
        )
        assert len(results) <= 3
        assert all(r.reduced_formula != query.reduced_formula for r in results)


class TestMPCBridge:
    def test_export_and_precomputed_retriever(self, tiny_test_cases, tmp_path):
        import numpy as np

        from src.evaluation.mpc_data import export_mpc_arrays
        from src.evaluation.mpc_retriever import PrecomputedEmbeddingRetriever

        meta = export_mpc_arrays(
            tiny_test_cases, {"val": tiny_test_cases[:2]},
            tmp_path / "arrays.npz",
        )
        blobs = np.load(tmp_path / "arrays.npz")
        assert blobs["train_comp"].shape == (
            len(tiny_test_cases), meta["n_elements"]
        )
        assert blobs["val_y"].shape == (2, meta["n_precursors"])

        rng = np.random.default_rng(0)
        emb = rng.normal(size=(len(tiny_test_cases), 8))
        query = tiny_test_cases[0]
        ret = PrecomputedEmbeddingRetriever(
            tiny_test_cases, emb, {query.reduced_formula: emb[0]}
        )
        results = ret.retrieve(query, k=3)
        assert len(results) == 3
        assert all(r.reduced_formula != query.reduced_formula for r in results)
        # unknown query embeddings yield no results rather than crashing
        assert ret.retrieve(tiny_test_cases[1], k=3) == []
