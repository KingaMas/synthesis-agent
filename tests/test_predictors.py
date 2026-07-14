"""
Tests for recipe predictors and the PlanningBenchmark harness.

All tests run offline (no API keys, no network).
"""

import pytest

from src.evaluation.planning import PlanningBenchmark, RecipePrediction
from src.evaluation.predictors import (
    CachedPredictor,
    RetrievalPredictor,
    RulePredictor,
    parse_prediction_json,
)


class MockPredictor:
    """Returns a fixed correct prediction for every case."""

    name = "mock"

    def __init__(self, fail_on: str = None):
        self.fail_on = fail_on
        self.calls = 0

    def predict(self, case):
        self.calls += 1
        if self.fail_on and case.reduced_formula == self.fail_on:
            raise RuntimeError("simulated API failure")
        from src.evaluation.planning import extract_ground_truth

        gt = extract_ground_truth(case.raw_recipe)
        return RecipePrediction(
            target_formula=case.reduced_formula,
            precursor_formulas=list(gt.precursor_formulas),
            method_family=gt.method_family,
            max_heating_temp_C=gt.max_heating_temp_C,
        )


class TestPlanningBenchmark:
    def test_perfect_predictor_scores_one(self, tiny_test_cases):
        bench = PlanningBenchmark(tiny_test_cases, verbose=False)
        results = bench.evaluate(MockPredictor(), name="mock")
        assert results.n_cases == len(tiny_test_cases)
        assert results.n_failures == 0
        assert all(s == 1.0 for s in results.per_case["element_jaccard"])
        assert all(s == 1.0 for s in results.per_case["formula_f1"])
        assert all(s == 1.0 for s in results.per_case["method_accuracy"])

    def test_one_failure_does_not_kill_run(self, tiny_test_cases):
        target = tiny_test_cases[0].reduced_formula
        bench = PlanningBenchmark(tiny_test_cases, verbose=False)
        results = bench.evaluate(MockPredictor(fail_on=target), name="mock")
        assert results.n_failures == 1
        assert results.n_cases == len(tiny_test_cases)
        # failed case scores zero on set metrics
        idx = [tc.reduced_formula for tc in tiny_test_cases].index(target)
        assert results.per_case["element_jaccard"][idx] == 0.0
        # temperature excluded, not zeroed
        assert results.per_case["temp_abs_err"][idx] is None

    def test_aggregate_structure(self, tiny_test_cases):
        bench = PlanningBenchmark(tiny_test_cases, verbose=False)
        agg = bench.evaluate(MockPredictor(), name="mock").aggregate()
        assert agg["metrics"]["element_jaccard"]["mean"] == 1.0
        assert "ci_lo" in agg["metrics"]["formula_f1"]
        assert "coverage_pred" in agg["temperature"]
        assert agg["latency"]["mean_s"] >= 0.0


class TestRulePredictor:
    def test_deterministic(self, tiny_test_cases):
        p1 = RulePredictor(tiny_test_cases)
        p2 = RulePredictor(tiny_test_cases)
        q = tiny_test_cases[0]
        assert p1.predict(q).precursor_formulas == p2.predict(q).precursor_formulas

    def test_precursors_cover_cation_elements(self, tiny_test_cases):
        predictor = RulePredictor(tiny_test_cases)
        for case in tiny_test_cases:
            pred = predictor.predict(case)
            covered = set(pred.precursor_elements)
            cations = set(case.elements) - {"O", "H", "C", "N"}
            assert cations <= covered, (
                f"{case.reduced_formula}: {cations - covered} not covered"
            )

    def test_majority_method_and_temp(self, tiny_test_cases):
        predictor = RulePredictor(tiny_test_cases)
        pred = predictor.predict(tiny_test_cases[0])
        # fixture corpus is majority solid-state
        assert pred.method_family == "solid-state"


class TestRetrievalPredictor:
    def test_copies_neighbor_recipe(self, tiny_test_cases):
        from src.evaluation.baselines import ElementJaccardRetriever

        retriever = ElementJaccardRetriever(corpus=tiny_test_cases)
        predictor = RetrievalPredictor(retriever, name="retrieval-jaccard")
        query = tiny_test_cases[0]
        pred = predictor.predict(query)
        assert pred.error is None
        assert pred.provenance is not None
        # never predicts from the query's own recipe
        assert pred.provenance != query.reduced_formula

    def test_empty_retrieval_is_failure(self, tiny_test_cases):
        class EmptyRetriever:
            def retrieve(self, query, k):
                return []

        predictor = RetrievalPredictor(EmptyRetriever(), name="empty")
        pred = predictor.predict(tiny_test_cases[0])
        assert pred.error == "no neighbor"


class TestCachedPredictor:
    def test_cache_hit_skips_inner(self, tiny_test_cases, tmp_path):
        inner = MockPredictor()
        cached = CachedPredictor(inner, cache_dir=tmp_path)
        case = tiny_test_cases[0]

        first = cached.predict(case)
        second = cached.predict(case)
        assert inner.calls == 1
        assert first.precursor_formulas == second.precursor_formulas

    def test_failure_cached_and_not_retried_by_default(
        self, tiny_test_cases, tmp_path
    ):
        case = tiny_test_cases[0]
        inner = MockPredictor(fail_on=case.reduced_formula)
        cached = CachedPredictor(inner, cache_dir=tmp_path, max_retries=2)

        pred = cached.predict(case)
        assert pred.error is not None
        calls_after_first = inner.calls  # == max_retries

        cached.predict(case)
        assert inner.calls == calls_after_first  # cache hit, no retry

    def test_retry_failures_reattempts(self, tiny_test_cases, tmp_path):
        case = tiny_test_cases[0]
        inner = MockPredictor(fail_on=case.reduced_formula)
        cached = CachedPredictor(inner, cache_dir=tmp_path, max_retries=1)
        cached.predict(case)

        inner.fail_on = None  # "API recovered"
        retry = CachedPredictor(
            inner, cache_dir=tmp_path, retry_failures=True, max_retries=1
        )
        pred = retry.predict(case)
        assert pred.error is None


class TestParsePredictionJson:
    GOOD = '{"precursors": ["BaCO3", "TiO2"], "method": "solid-state", "max_heating_temperature_C": 1100, "rationale": "standard route"}'

    def test_clean_json(self):
        pred = parse_prediction_json(self.GOOD, "BaTiO3")
        assert pred.error is None
        assert pred.precursor_formulas == ["BaCO3", "TiO2"]
        assert pred.method_family == "solid-state"
        assert pred.max_heating_temp_C == 1100.0

    def test_fenced_json(self):
        pred = parse_prediction_json(f"```json\n{self.GOOD}\n```", "BaTiO3")
        assert pred.error is None
        assert pred.method_family == "solid-state"

    def test_json_embedded_in_prose(self):
        text = f"Here is my analysis of BaTiO3.\n\nFinal answer:\n{self.GOOD}\n\nGood luck!"
        pred = parse_prediction_json(text, "BaTiO3")
        assert pred.error is None
        assert pred.precursor_formulas == ["BaCO3", "TiO2"]

    def test_garbage_preserves_raw_text(self):
        pred = parse_prediction_json("I cannot help with that.", "BaTiO3")
        assert pred.error == "parse"
        assert pred.raw_text == "I cannot help with that."

    def test_free_text_method_normalized(self):
        text = '{"precursors": ["TiO2"], "method": "conventional ceramic route with sintering", "max_heating_temperature_C": null}'
        pred = parse_prediction_json(text, "BaTiO3")
        assert pred.method_family == "solid-state"
        assert pred.max_heating_temp_C is None

    def test_non_numeric_temperature_dropped(self):
        text = '{"precursors": ["TiO2"], "method": "sol-gel", "max_heating_temperature_C": "around 900"}'
        pred = parse_prediction_json(text, "BaTiO3")
        assert pred.error is None
        assert pred.max_heating_temp_C is None
