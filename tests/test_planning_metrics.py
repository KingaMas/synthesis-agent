"""
Tests for planning-benchmark ground-truth extraction and metrics.

All tests are offline and use hand-built recipe dicts with known answers.
"""

import pytest

from src.evaluation.planning import (
    RecipeGroundTruth,
    RecipePrediction,
    extract_ground_truth,
    extract_max_heating_temp_C,
    method_accuracy,
    normalize_method,
    precursor_element_jaccard,
    precursor_formula_f1,
    temperature_abs_error,
)


def _heating_op(values, units="°C"):
    return {
        "type": "HeatingOperation",
        "token": "heated",
        "conditions": {
            "heating_temperature": [
                {"min_value": min(values), "max_value": max(values),
                 "values": values, "units": units}
            ],
            "heating_time": [],
            "heating_atmosphere": [],
        },
    }


class TestExtractMaxHeatingTemp:
    def test_celsius_entry(self):
        recipe = {"operations": [_heating_op([900.0])]}
        assert extract_max_heating_temp_C(recipe) == 900.0

    def test_kelvin_converted(self):
        recipe = {"operations": [_heating_op([1273.15], units="K")]}
        assert extract_max_heating_temp_C(recipe) == pytest.approx(1000.0)

    def test_max_across_mixed_units(self):
        # 1373.15 K = 1100 °C should beat the 1000 °C entry
        recipe = {
            "operations": [
                _heating_op([1000.0]),
                _heating_op([1373.15], units="K"),
            ]
        }
        assert extract_max_heating_temp_C(recipe) == pytest.approx(1100.0)

    def test_bare_c_unit_treated_as_celsius(self):
        recipe = {"operations": [_heating_op([850.0], units="C")]}
        assert extract_max_heating_temp_C(recipe) == 850.0

    def test_empty_operations_returns_none(self):
        assert extract_max_heating_temp_C({"operations": []}) is None
        assert extract_max_heating_temp_C({}) is None

    def test_falls_back_to_max_value_when_values_empty(self):
        op = _heating_op([700.0])
        op["conditions"]["heating_temperature"][0]["values"] = []
        assert extract_max_heating_temp_C({"operations": [op]}) == 700.0


class TestExtractGroundTruth:
    def test_full_recipe(self):
        recipe = {
            "synthesis_type": "solid-state",
            "precursors_formula_s": ["BaCO3", "TiO2"],
            "operations": [_heating_op([1200.0])],
        }
        gt = extract_ground_truth(recipe)
        assert gt.method_family == "solid-state"
        assert set(gt.precursor_formulas) == {"BaCO3", "TiO2"}
        assert set(gt.precursor_elements) == {"Ba", "C", "O", "Ti"}
        assert gt.max_heating_temp_C == 1200.0

    def test_falls_back_to_precursor_dicts(self):
        recipe = {
            "synthesis_type": "sol-gel",
            "precursors": [{"material_string": "Fe2O3"}],
        }
        gt = extract_ground_truth(recipe)
        assert gt.precursor_formulas == ["Fe2O3"]
        assert gt.max_heating_temp_C is None


class TestNormalizeMethod:
    def test_exact_labels(self):
        assert normalize_method("solid-state") == "solid-state"
        assert normalize_method("sol-gel") == "sol-gel"

    def test_free_text_maps_to_family(self):
        assert normalize_method("calcined the pellet at 900 C") == "solid-state"
        assert normalize_method("a sol gel route with gelation") == "sol-gel"

    def test_unknown_is_other(self):
        assert normalize_method("electrodeposition") == "other"
        assert normalize_method("") == "other"


GT = RecipeGroundTruth(
    precursor_formulas=["BaCO3", "TiO2"],
    precursor_elements=["Ba", "C", "O", "Ti"],
    method_family="solid-state",
    max_heating_temp_C=1200.0,
)


class TestPrecursorMetrics:
    def test_element_jaccard_exact(self):
        pred = RecipePrediction("BaTiO3", precursor_formulas=["BaCO3", "TiO2"])
        assert precursor_element_jaccard(pred, GT) == 1.0

    def test_element_jaccard_partial(self):
        # {Ti, O} vs {Ba, C, O, Ti} -> 2/4
        pred = RecipePrediction("BaTiO3", precursor_formulas=["TiO2"])
        assert precursor_element_jaccard(pred, GT) == 0.5

    def test_failed_prediction_scores_zero(self):
        pred = RecipePrediction("BaTiO3", error="API timeout")
        assert precursor_element_jaccard(pred, GT) == 0.0
        assert precursor_formula_f1(pred, GT) == 0.0

    def test_formula_f1_exact(self):
        pred = RecipePrediction("BaTiO3", precursor_formulas=["TiO2", "BaCO3"])
        assert precursor_formula_f1(pred, GT) == 1.0

    def test_formula_f1_half_overlap(self):
        # pred {BaCO3, SrCO3}: P=1/2, R=1/2 -> F1=0.5
        pred = RecipePrediction("BaTiO3", precursor_formulas=["BaCO3", "SrCO3"])
        assert precursor_formula_f1(pred, GT) == pytest.approx(0.5)

    def test_formula_f1_both_empty(self):
        empty_gt = RecipeGroundTruth([], [], "solid-state", None)
        pred = RecipePrediction("BaTiO3")
        assert precursor_formula_f1(pred, empty_gt) == 1.0

    def test_formula_f1_one_empty(self):
        pred = RecipePrediction("BaTiO3")
        assert precursor_formula_f1(pred, GT) == 0.0

    def test_unparseable_formulas_dropped(self):
        pred = RecipePrediction(
            "BaTiO3", precursor_formulas=["BaCO3", "TiO2", "not a formula!!"]
        )
        assert precursor_formula_f1(pred, GT) == 1.0

    def test_normalization_matches_equivalent_formulas(self):
        # Ba2C2O6 reduces to BaCO3
        pred = RecipePrediction("BaTiO3", precursor_formulas=["Ba2C2O6", "TiO2"])
        assert precursor_formula_f1(pred, GT) == 1.0

    def test_hydrate_does_not_match_anhydrous(self):
        gt = RecipeGroundTruth(
            ["Fe(NO3)3"], ["Fe", "N", "O"], "sol-gel", None
        )
        pred = RecipePrediction("Fe2O3", precursor_formulas=["Fe(NO3)3.9H2O"])
        assert precursor_formula_f1(pred, gt) == 0.0


class TestMethodAccuracy:
    def test_match(self):
        pred = RecipePrediction("BaTiO3", method_family="solid-state")
        assert method_accuracy(pred, GT) == 1.0

    def test_free_text_prediction_mapped(self):
        pred = RecipePrediction(
            "BaTiO3", method_family="conventional ceramic sintering route"
        )
        assert method_accuracy(pred, GT) == 1.0

    def test_other_scores_zero(self):
        pred = RecipePrediction("BaTiO3", method_family="hydrothermal")
        assert method_accuracy(pred, GT) == 0.0
        pred = RecipePrediction("BaTiO3", method_family="")
        assert method_accuracy(pred, GT) == 0.0


class TestTemperatureError:
    def test_basic_error(self):
        pred = RecipePrediction("BaTiO3", max_heating_temp_C=1100.0)
        assert temperature_abs_error(pred, GT) == 100.0

    def test_missing_gt_returns_none(self):
        gt = RecipeGroundTruth(["BaCO3"], ["Ba", "C", "O"], "solid-state", None)
        pred = RecipePrediction("BaTiO3", max_heating_temp_C=1100.0)
        assert temperature_abs_error(pred, gt) is None

    def test_missing_prediction_returns_none(self):
        pred = RecipePrediction("BaTiO3")
        assert temperature_abs_error(pred, GT) is None

    def test_out_of_range_prediction_returns_none(self):
        pred = RecipePrediction("BaTiO3", max_heating_temp_C=99999.0)
        assert temperature_abs_error(pred, GT) is None
        pred = RecipePrediction("BaTiO3", max_heating_temp_C=-5.0)
        assert temperature_abs_error(pred, GT) is None


class TestPredictionPostInit:
    def test_elements_derived_from_formulas(self):
        pred = RecipePrediction("BaTiO3", precursor_formulas=["BaCO3", "TiO2"])
        assert pred.precursor_elements == ["Ba", "C", "O", "Ti"]

    def test_explicit_elements_kept(self):
        pred = RecipePrediction(
            "BaTiO3", precursor_formulas=["BaCO3"], precursor_elements=["Ba"]
        )
        assert pred.precursor_elements == ["Ba"]
