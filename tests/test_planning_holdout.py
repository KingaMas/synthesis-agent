"""
Tests for the held-out planning test set, leakage guard, and RecipeStore.

Uses a synthetic mini recipe corpus in the REAL corpus format (list of
dicts with target.material_string / targets_formula_s / synthesis_type),
gzipped into a temp file so load_recipes() reads it like the real asset.
"""

import gzip
import json

import pytest

from src.evaluation.planning import (
    build_planning_test_set,
    split_heldout,
)
from src.evaluation.predictors import RecipeStore


def _recipe(target, precursors, synthesis_type="solid-state", temp_c=1000.0):
    return {
        "target": {"material_string": target},
        "targets_formula_s": [target],
        "precursors_formula_s": precursors,
        "precursors": [{"material_string": p} for p in precursors],
        "synthesis_type": synthesis_type,
        "paragraph_string": f"{target} was synthesized from {', '.join(precursors)}.",
        "operations": [
            {
                "type": "HeatingOperation",
                "token": "calcined",
                "conditions": {
                    "heating_temperature": [
                        {"min_value": temp_c, "max_value": temp_c,
                         "values": [temp_c], "units": "°C"}
                    ],
                },
            }
        ],
        "doi": "10.0000/test",
    }


MINI_RECIPES = [
    # 4 solid-state
    _recipe("BaTiO3", ["BaCO3", "TiO2"]),
    _recipe("LiCoO2", ["Li2CO3", "Co3O4"]),
    _recipe("SrTiO3", ["SrCO3", "TiO2"]),
    _recipe("NaFeO2", ["Na2CO3", "Fe2O3"]),
    # 4 sol-gel
    _recipe("ZnAl2O4", ["Zn(NO3)2", "Al(NO3)3"], synthesis_type="sol-gel"),
    _recipe("LaAlO3", ["La(NO3)3", "Al(NO3)3"], synthesis_type="sol-gel"),
    _recipe("YAlO3", ["Y(NO3)3", "Al(NO3)3"], synthesis_type="sol-gel"),
    _recipe("MgAl2O4", ["Mg(NO3)2", "Al(NO3)3"], synthesis_type="sol-gel"),
    # duplicate formula (must be deduplicated)
    _recipe("BaTiO3", ["BaO", "TiO2"]),
    # unusable cases (must be filtered out)
    _recipe("not a formula !!", ["BaCO3"]),
    _recipe("Fe2O3", []),                                    # no precursor GT
    _recipe("CuO", ["Cu(NO3)2"], synthesis_type="hydrothermal"),  # unknown family
]


@pytest.fixture(scope="module")
def mini_recipes_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("recipes") / "mini_recipes.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(MINI_RECIPES, fh)
    return path


class TestBuildPlanningTestSet:
    def test_stratified_50_50(self, mini_recipes_path):
        cases = build_planning_test_set(mini_recipes_path, n_cases=8, seed=42)
        methods = [tc.synthesis_method for tc in cases]
        assert methods.count("solid-state") == 4
        assert methods.count("sol-gel") == 4

    def test_seed_reproducible(self, mini_recipes_path):
        a = build_planning_test_set(mini_recipes_path, n_cases=6, seed=7)
        b = build_planning_test_set(mini_recipes_path, n_cases=6, seed=7)
        assert [tc.reduced_formula for tc in a] == [tc.reduced_formula for tc in b]

    def test_unusable_recipes_filtered(self, mini_recipes_path):
        cases = build_planning_test_set(mini_recipes_path, n_cases=20, seed=42)
        formulas = {tc.reduced_formula for tc in cases}
        assert "Fe2O3" not in formulas   # no precursor ground truth
        assert "CuO" not in formulas     # method family not in KNOWN_METHODS
        # duplicate BaTiO3 appears exactly once
        assert [tc.reduced_formula for tc in cases].count("BaTiO3") <= 1

    def test_every_case_has_ground_truth(self, mini_recipes_path):
        from src.evaluation.planning import KNOWN_METHODS, extract_ground_truth

        cases = build_planning_test_set(mini_recipes_path, n_cases=8, seed=42)
        assert cases
        for tc in cases:
            gt = extract_ground_truth(tc.raw_recipe)
            assert gt.precursor_formulas
            assert gt.method_family in KNOWN_METHODS


class TestSplitHeldout:
    def test_zero_overlap(self, mini_recipes_path):
        from src.evaluation.test_set_builder import build_retrieval_corpus

        corpus = build_retrieval_corpus(mini_recipes_path)
        test_set = build_planning_test_set(mini_recipes_path, n_cases=4, seed=42)
        train, heldout = split_heldout(corpus, test_set)

        train_formulas = {tc.reduced_formula for tc in train}
        assert not (train_formulas & heldout)
        assert heldout == {tc.reduced_formula for tc in test_set}
        assert len(train) == len(corpus) - len(
            [tc for tc in corpus if tc.reduced_formula in heldout]
        )


class TestRecipeStore:
    def test_lookup_by_reduced_formula(self, mini_recipes_path):
        store = RecipeStore(mini_recipes_path)
        recipes = store.lookup("BaTiO3")
        assert len(recipes) == 2  # both BaTiO3 recipes indexed

    def test_lookup_normalizes_formula(self, mini_recipes_path):
        store = RecipeStore(mini_recipes_path)
        # Ba2Ti2O6 reduces to BaTiO3
        assert store.lookup("Ba2Ti2O6") == store.lookup("BaTiO3")

    def test_excluded_formula_never_returned(self, mini_recipes_path):
        store = RecipeStore(
            mini_recipes_path, exclude_formulas=frozenset({"BaTiO3", "LiCoO2"})
        )
        assert store.lookup("BaTiO3") == []
        assert store.lookup("LiCoO2") == []
        assert store.lookup("SrTiO3")  # non-excluded still present

    def test_unparseable_lookup_returns_empty(self, mini_recipes_path):
        store = RecipeStore(mini_recipes_path)
        assert store.lookup("!!!") == []
