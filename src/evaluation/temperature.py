"""
Temperature protocol reconciliation (T5).

Prein et al. (2025) report calcination/sintering MAE < 126 degC with a
per-operation protocol on solid-state recipes; our planning benchmark
scores a single max-heating temperature across mixed routes and lands
above 310 degC. This module separates the three confounded factors:

1. protocol      — per-operation targets (calcination temp, sintering
                   temp, from operation tokens) vs max over all heating
                   operations;
2. label noise   — the corpus itself disagrees: different papers report
                   different temperatures for the same material. The
                   leave-one-out replication MAE across duplicate
                   (formula, solid-state) recipes is the floor no
                   predictor can beat on this ground truth;
3. prediction    — rule and retrieval-copy MAEs re-scored under both
                   protocols on identical support sets.

Ground truth comes from the structured `operations` field (Kononova et
al. extraction), the same source as extract_max_heating_temp_C.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from src.evaluation.planning import TEMP_MAX_C, TEMP_MIN_C
from src.evaluation.test_set_builder import TestCase

# Operation-token classes for the per-operation protocol. Tokens are the
# lemmas observed in the corpus (top heating-op tokens); "calcined" and
# "sintered" dominate. Drying is excluded on purpose: Prein et al. score
# calcination and sintering.
CALCINATION_TOKENS = frozenset(
    {"calcined", "calcination", "calcining", "calcine",
     "precalcined", "pre-calcined", "precalcination"}
)
SINTERING_TOKENS = frozenset({"sintered", "sintering", "sinter"})

PROTOCOLS = ("max_heating", "calcination", "sintering")


def _entry_temps_C(conditions: dict) -> list[float]:
    """Heating temperatures in degC, restricted to the plausibility
    window (TEMP_MIN_C, TEMP_MAX_C] that planning.py applies to
    predictions. The corpus ground truth contains extraction artifacts
    — concatenated digit strings like 9001150 for "900, 1150 degC" —
    which the max-over-operations protocol would otherwise surface.
    """
    temps: list[float] = []
    for entry in conditions.get("heating_temperature") or []:
        values = entry.get("values") or []
        if not values and entry.get("max_value") is not None:
            values = [entry["max_value"]]
        units = (entry.get("units") or "°C").strip()
        for v in values:
            if v is None:
                continue
            t = float(v) - 273.15 if units == "K" else float(v)
            if TEMP_MIN_C < t <= TEMP_MAX_C:
                temps.append(t)
    return temps


def extract_operation_temp_C(recipe: dict, tokens: frozenset) -> Optional[float]:
    """Max heating temperature among operations whose token is in `tokens`."""
    temps: list[float] = []
    for op in recipe.get("operations") or []:
        if (op.get("token") or "").lower() not in tokens:
            continue
        temps.extend(_entry_temps_C(op.get("conditions") or {}))
    return max(temps) if temps else None


def protocol_temp_C(recipe: dict, protocol: str) -> Optional[float]:
    if protocol == "max_heating":
        temps = [t for op in recipe.get("operations") or []
                 for t in _entry_temps_C(op.get("conditions") or {})]
        return max(temps) if temps else None
    if protocol == "calcination":
        return extract_operation_temp_C(recipe, CALCINATION_TOKENS)
    if protocol == "sintering":
        return extract_operation_temp_C(recipe, SINTERING_TOKENS)
    raise ValueError(protocol)


def replication_noise(
    recipes_by_formula: dict[str, list[dict]], protocol: str
) -> dict:
    """Leave-one-out replication MAE across duplicate-formula recipes.

    For each formula with >= 2 recipes reporting the protocol's
    temperature, predict each recipe's value with the mean of the
    others. This is the error an oracle that read every OTHER paper
    about the same material would still make — the ground-truth noise
    floor for this protocol. Also reports the mean absolute pairwise
    difference for reference.
    """
    loo_errors: list[float] = []
    pair_diffs: list[float] = []
    n_formulas = 0
    for recs in recipes_by_formula.values():
        temps = [t for t in (protocol_temp_C(r, protocol) for r in recs)
                 if t is not None]
        if len(temps) < 2:
            continue
        n_formulas += 1
        arr = np.asarray(temps)
        total = arr.sum()
        loo_errors.extend(
            abs(t - (total - t) / (len(arr) - 1)) for t in arr
        )
        pair_diffs.extend(
            abs(a - b) for i, a in enumerate(arr) for b in arr[i + 1:]
        )
    return {
        "loo_mae": float(np.mean(loo_errors)) if loo_errors else None,
        "pairwise_mad": float(np.mean(pair_diffs)) if pair_diffs else None,
        "n_formulas": n_formulas,
        "n_reports": len(loo_errors),
    }


class ProtocolRulePredictor:
    """Train-median temperature per protocol (solid-state fit corpus)."""

    def __init__(self, fit_corpus: list[TestCase]):
        self._median: dict[str, Optional[float]] = {}
        for protocol in PROTOCOLS:
            temps = [t for t in (protocol_temp_C(tc.raw_recipe, protocol)
                                 for tc in fit_corpus) if t is not None]
            self._median[protocol] = float(np.median(temps)) if temps else None

    def predict(self, query: TestCase, protocol: str) -> Optional[float]:
        return self._median[protocol]


class ProtocolRetrievalCopyPredictor:
    """Copy the protocol temperature of the nearest neighbour that has one.

    Mirrors the planning benchmark's RetrievalPredictor (copy the
    retrieved recipe's ground truth), extended with a fixed candidate
    budget: the first of the top-`budget` neighbours reporting the
    protocol's temperature is used.
    """

    def __init__(self, retriever, budget: int = 5):
        self.retriever = retriever
        self.budget = budget

    def predict(self, query: TestCase, protocol: str) -> Optional[float]:
        for tc in self.retriever.retrieve(query, k=self.budget):
            t = protocol_temp_C(tc.raw_recipe, protocol)
            if t is not None:
                return t
        return None


def evaluate_protocols(
    predictors: dict[str, object],
    queries: list[TestCase],
) -> dict:
    """MAE per (predictor, protocol) on per-protocol and strict common support.

    Per-protocol support: queries whose ground truth exists for that
    protocol AND for which every predictor produced a value (audit rule:
    method comparisons need identical support). Strict support
    additionally requires ground truth under ALL protocols, so protocol
    effects can be compared on one fixed query set.
    """
    gt = {
        protocol: [protocol_temp_C(q.raw_recipe, protocol) for q in queries]
        for protocol in PROTOCOLS
    }
    preds = {
        (name, protocol): [p.predict(q, protocol) for q in queries]
        for name, p in predictors.items()
        for protocol in PROTOCOLS
    }

    def support(protocol: str, strict: bool) -> list[int]:
        idx = []
        for i in range(len(queries)):
            gts = [gt[pr][i] for pr in PROTOCOLS] if strict else [gt[protocol][i]]
            if any(g is None for g in gts):
                continue
            if any(preds[(n, protocol)][i] is None for n in predictors):
                continue
            idx.append(i)
        return idx

    out: dict = {}
    for strict in (False, True):
        key = "strict_common_support" if strict else "per_protocol_support"
        out[key] = {}
        for protocol in PROTOCOLS:
            idx = support(protocol, strict)
            entry: dict = {"n": len(idx)}
            for name in predictors:
                errors = [
                    abs(preds[(name, protocol)][i] - gt[protocol][i])
                    for i in idx
                ]
                entry[name] = {
                    "mae": float(np.mean(errors)) if errors else None,
                    "median_ae": float(np.median(errors)) if errors else None,
                }
            out[key][protocol] = entry
    return out
