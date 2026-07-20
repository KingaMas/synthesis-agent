"""Tests for the T5 temperature-protocol reconciliation module."""

import pytest

from src.evaluation.temperature import (
    CALCINATION_TOKENS,
    SINTERING_TOKENS,
    protocol_temp_C,
    replication_noise,
)


def _recipe(ops):
    return {"operations": [
        {"token": tok,
         "conditions": {"heating_temperature": [
             {"values": vals, "units": units}]}}
        for tok, vals, units in ops
    ]}


class TestProtocolExtraction:
    def test_per_operation_protocols_split_by_token(self):
        r = _recipe([("calcined", [900.0], "°C"),
                     ("sintered", [1400.0], "°C")])
        assert protocol_temp_C(r, "calcination") == 900.0
        assert protocol_temp_C(r, "sintering") == 1400.0
        assert protocol_temp_C(r, "max_heating") == 1400.0

    def test_kelvin_conversion(self):
        r = _recipe([("calcined", [1173.15], "K")])
        assert protocol_temp_C(r, "calcination") == pytest.approx(900.0)

    def test_missing_operation_returns_none(self):
        r = _recipe([("dried", [120.0], "°C")])
        assert protocol_temp_C(r, "calcination") is None
        assert protocol_temp_C(r, "sintering") is None
        assert protocol_temp_C(r, "max_heating") == 120.0

    def test_concatenated_digit_artifacts_are_dropped(self):
        # "900, 1150 degC" mined as the single value 9001150
        r = _recipe([("calcined", [9001150.0], "°C"),
                     ("sintered", [1400.0], "°C")])
        assert protocol_temp_C(r, "calcination") is None
        assert protocol_temp_C(r, "max_heating") == 1400.0

    def test_token_classes_are_disjoint(self):
        assert not CALCINATION_TOKENS & SINTERING_TOKENS


class TestReplicationNoise:
    def test_loo_mae_over_duplicate_formulas(self):
        by_formula = {
            "BaTiO3": [_recipe([("calcined", [900.0], "°C")]),
                       _recipe([("calcined", [1000.0], "°C")])],
            "SrTiO3": [_recipe([("calcined", [800.0], "°C")])],  # singleton
        }
        r = replication_noise(by_formula, "calcination")
        assert r["n_formulas"] == 1
        assert r["n_reports"] == 2
        assert r["loo_mae"] == pytest.approx(100.0)
        assert r["pairwise_mad"] == pytest.approx(100.0)
