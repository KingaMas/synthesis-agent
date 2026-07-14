"""
Synthesis planning benchmark for SKY.

Evaluates methods that PREDICT a synthesis recipe (precursors, method
family, heating temperature) for a target formula, against held-out
ground-truth recipes from mp_synthesis_recipes.json.gz.

Unlike the retrieval benchmark (benchmark.py), which scores whether
retrieved *neighbor materials* have transferable recipes, this benchmark
scores an explicit per-target recipe prediction.

Metrics
-------
element_jaccard   Jaccard similarity of precursor element sets
                  (predicted vs ground truth). Failed predictions score 0.
formula_f1        Set F1 over pymatgen-normalized precursor reduced
                  formulas.
method_accuracy   Exact match of canonical method family
                  (2-class on this corpus: solid-state / sol-gel).
temp_abs_error    |predicted - true| max heating temperature in °C.
                  None (excluded from MAE) when either side is missing.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from pymatgen.core import Composition

from src.evaluation.benchmark import _jaccard
from src.evaluation.statistics import bootstrap_ci
from src.evaluation.test_set_builder import (
    TestCase,
    _extract_precursor_elements,
    classify_synthesis_method,
    load_recipes,
)


# Canonical method labels as they appear in recipe["synthesis_type"]
KNOWN_METHODS = ("solid-state", "sol-gel")

# Predicted temperatures outside this range (°C) are treated as parse failures
TEMP_MIN_C = 0.0
TEMP_MAX_C = 3000.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RecipePrediction:
    """A predicted synthesis recipe for one target material."""

    target_formula: str
    precursor_formulas: list[str] = field(default_factory=list)
    precursor_elements: list[str] = field(default_factory=list)
    method_family: str = "other"
    max_heating_temp_C: Optional[float] = None
    raw_text: str = ""
    provenance: Optional[str] = None   # e.g. source neighbor formula
    error: Optional[str] = None        # non-None => failed prediction
    latency_s: float = 0.0
    model: Optional[str] = None

    def __post_init__(self):
        # Derive elements from formulas when not supplied explicitly
        if not self.precursor_elements and self.precursor_formulas:
            elements: set[str] = set()
            for f in self.precursor_formulas:
                try:
                    elements.update(str(el) for el in Composition(f).elements)
                except Exception:
                    pass
            self.precursor_elements = sorted(elements)


@dataclass
class RecipeGroundTruth:
    """Ground-truth recipe fields extracted from a raw recipe dict."""

    precursor_formulas: list[str]
    precursor_elements: list[str]
    method_family: str
    max_heating_temp_C: Optional[float]


class RecipePredictor(Protocol):
    """Common interface for all recipe prediction methods."""

    name: str

    def predict(self, case: TestCase) -> RecipePrediction:
        """Predict a recipe for *case*. Must not raise for routine failures."""
        ...


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------

def extract_max_heating_temp_C(recipe: dict) -> Optional[float]:
    """Max heating temperature across all operations, converted to °C.

    Units observed in the corpus: '°C' (72,516), 'K' (9,025), 'C' (87).
    Returns None when no heating temperature is recorded (~32% of recipes).
    """
    temps_c: list[float] = []
    for op in recipe.get("operations") or []:
        conditions = op.get("conditions") or {}
        for entry in conditions.get("heating_temperature") or []:
            values = entry.get("values") or []
            if not values and entry.get("max_value") is not None:
                values = [entry["max_value"]]
            units = (entry.get("units") or "°C").strip()
            for v in values:
                if v is None:
                    continue
                temps_c.append(float(v) - 273.15 if units == "K" else float(v))
    return max(temps_c) if temps_c else None


def normalize_formula_set(formulas: list[str]) -> set[str]:
    """Normalize formulas to pymatgen reduced form; drop unparseable ones."""
    normalized: set[str] = set()
    for f in formulas:
        try:
            normalized.add(Composition(f).reduced_formula)
        except Exception:
            continue
    return normalized


def normalize_method(text: str) -> str:
    """Map a method string to a canonical family.

    Exact synthesis_type labels pass through; free text (e.g. LLM output
    like "conventional solid state reaction") falls back to keyword
    classification.
    """
    if not text:
        return "other"
    cleaned = text.strip().lower()
    for method in KNOWN_METHODS:
        if cleaned == method:
            return method
    return classify_synthesis_method(cleaned)


def _extract_precursor_formulas(recipe: dict) -> list[str]:
    """Precursor formula strings, preferring the flat precursors_formula_s."""
    formulas = recipe.get("precursors_formula_s") or []
    if not formulas:
        formulas = [
            p.get("material_string") or p.get("material_formula") or ""
            for p in recipe.get("precursors") or []
        ]
    return [f for f in formulas if f]


def extract_ground_truth(recipe: dict) -> RecipeGroundTruth:
    """Build ground truth from a raw recipe dict.

    Uses the clean recipe["synthesis_type"] label directly rather than
    classify_synthesis_method(), which is order-biased on mixed text.
    """
    formulas = _extract_precursor_formulas(recipe)
    elements: set[str] = set()
    for f in formulas:
        try:
            elements.update(str(el) for el in Composition(f).elements)
        except Exception:
            continue
    return RecipeGroundTruth(
        precursor_formulas=formulas,
        precursor_elements=sorted(elements),
        method_family=normalize_method(recipe.get("synthesis_type", "")),
        max_heating_temp_C=extract_max_heating_temp_C(recipe),
    )


# ---------------------------------------------------------------------------
# Held-out test set
# ---------------------------------------------------------------------------

def build_planning_test_set(
    recipes_path: Optional[Path] = None,
    n_cases: int = 100,
    seed: int = 42,
) -> list[TestCase]:
    """Build the held-out planning test set.

    Deduplicates by reduced formula, keeps only cases with usable ground
    truth (parseable precursors AND a known synthesis_type label), and
    stratifies 50/50 across the two method families so the majority-class
    baseline sits at 50% rather than the corpus prior (~86% solid-state).

    Deterministic for a given seed: candidates are sorted by reduced
    formula before the seeded shuffle, so results do not depend on file
    or dict ordering.
    """
    rng = random.Random(seed)
    recipes = load_recipes(recipes_path)

    seen: dict[str, TestCase] = {}
    for recipe in recipes:
        target = recipe.get("target", {}).get("material_string", "") or ""
        if not target:
            continue
        try:
            comp = Composition(target)
            reduced = comp.reduced_formula
        except Exception:
            continue
        if reduced in seen:
            continue

        method = normalize_method(recipe.get("synthesis_type", ""))
        if method not in KNOWN_METHODS:
            continue
        gt = extract_ground_truth(recipe)
        if not normalize_formula_set(gt.precursor_formulas):
            continue

        seen[reduced] = TestCase(
            material_id=recipe.get("target_id", ""),
            formula=target,
            reduced_formula=reduced,
            elements=sorted(str(el) for el in comp.elements),
            synthesis_method=method,
            precursor_elements=_extract_precursor_elements(recipe),
            raw_recipe=recipe,
        )

    by_method: dict[str, list[TestCase]] = {m: [] for m in KNOWN_METHODS}
    for tc in seen.values():
        by_method[tc.synthesis_method].append(tc)

    per_method = n_cases // len(KNOWN_METHODS)
    test_cases: list[TestCase] = []
    for method in KNOWN_METHODS:
        candidates = sorted(by_method[method], key=lambda tc: tc.reduced_formula)
        rng.shuffle(candidates)
        test_cases.extend(candidates[:per_method])

    rng.shuffle(test_cases)
    return test_cases


def load_planning_test_set(
    path: Optional[Path] = None,
    recipes_path: Optional[Path] = None,
    verify_against_builder: bool = False,
) -> list[TestCase]:
    """Load the committed, authoritative planning test set.

    The test set is a FILE, not a seed: build_planning_test_set() draws
    from the pool of pymatgen-parseable formulas, so a pymatgen upgrade
    reshuffles the entire seeded sample (verified: only 17/100 targets
    recovered under pymatgen 2026.5.4 vs the lockfile version). All
    evaluation must load this file.

    Each case is re-anchored to its corpus recipe by (reduced formula,
    DOI); raises if any case cannot be recovered.

    Args:
        path: Test-set JSON (default results/test_set_planning_seed42.json).
        recipes_path: Override recipe corpus path.
        verify_against_builder: If True, additionally assert that
            build_planning_test_set() reproduces the file exactly (only
            meaningful under the pinned environment; fails loudly on drift).
    """
    from src import PROJECT_ROOT

    if path is None:
        path = PROJECT_ROOT / "results" / "test_set_planning_seed42.json"
    spec = json.loads(Path(path).read_text())
    meta, records = spec["_meta"], spec["cases"]

    by_key: dict[tuple[str, str], dict] = {}
    for recipe in load_recipes(recipes_path):
        target = (recipe.get("target") or {}).get("material_string", "") or ""
        if not target:
            continue
        try:
            reduced = Composition(target).reduced_formula
        except Exception:
            continue
        key = (reduced, recipe.get("doi") or "")
        by_key.setdefault(key, recipe)

    cases: list[TestCase] = []
    missing: list[str] = []
    for rec in records:
        recipe = by_key.get((rec["reduced_formula"], rec["doi"] or ""))
        if recipe is None:
            missing.append(rec["reduced_formula"])
            continue
        comp = Composition(rec["reduced_formula"])
        cases.append(
            TestCase(
                material_id="",
                formula=rec["formula"],
                reduced_formula=rec["reduced_formula"],
                elements=sorted(str(el) for el in comp.elements),
                synthesis_method=rec["synthesis_method"],
                precursor_elements=_extract_precursor_elements(recipe),
                raw_recipe=recipe,
            )
        )
    if missing:
        raise RuntimeError(
            f"{len(missing)} test cases could not be re-anchored to the recipe "
            f"corpus (e.g. {missing[:3]}). The corpus file or formula parsing "
            f"has changed; do NOT silently rebuild the test set."
        )

    if verify_against_builder:
        rebuilt = build_planning_test_set(
            recipes_path, n_cases=meta["n_cases"], seed=meta["seed"]
        )
        rebuilt_f = sorted(tc.reduced_formula for tc in rebuilt)
        file_f = sorted(tc.reduced_formula for tc in cases)
        if rebuilt_f != file_f:
            diff = len(set(rebuilt_f) ^ set(file_f)) // 2
            raise RuntimeError(
                f"build_planning_test_set(seed={meta['seed']}) no longer "
                f"reproduces the committed test set ({diff} cases differ). "
                f"Environment has drifted from the pinned lockfile."
            )

    return cases


def split_heldout(
    corpus: list[TestCase], test_set: list[TestCase]
) -> tuple[list[TestCase], set[str]]:
    """Remove held-out formulas from a retrieval corpus.

    Leakage guard: exclusion happens at the recipe-corpus level (the H5
    embedding index carries no recipe information, so retrieving a
    held-out composition is harmless as long as its recipe is
    unreachable).

    Returns:
        (train_corpus, heldout_formulas)
    """
    heldout = {tc.reduced_formula for tc in test_set}
    train_corpus = [tc for tc in corpus if tc.reduced_formula not in heldout]
    return train_corpus, heldout


# ---------------------------------------------------------------------------
# Per-case metrics
# ---------------------------------------------------------------------------

def precursor_element_jaccard(
    pred: RecipePrediction, gt: RecipeGroundTruth
) -> float:
    """Jaccard similarity of precursor element sets; 0 for failed predictions."""
    if pred.error is not None and not pred.precursor_formulas:
        return 0.0
    return _jaccard(set(pred.precursor_elements), set(gt.precursor_elements))


def precursor_formula_f1(pred: RecipePrediction, gt: RecipeGroundTruth) -> float:
    """Set F1 over normalized precursor reduced formulas.

    Hydrate/solution-form mismatches (e.g. Fe(NO3)3·9H2O vs Fe(NO3)3)
    normalize to different reduced formulas and count as misses; this
    penalty applies identically to all methods.
    """
    if pred.error is not None and not pred.precursor_formulas:
        return 0.0
    pred_set = normalize_formula_set(pred.precursor_formulas)
    gt_set = normalize_formula_set(gt.precursor_formulas)
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def method_accuracy(pred: RecipePrediction, gt: RecipeGroundTruth) -> float:
    """1.0 iff canonical method families match.

    A predicted "other" always scores 0 on this corpus (ground truth is
    always solid-state or sol-gel).
    """
    if pred.error is not None:
        return 0.0
    return 1.0 if normalize_method(pred.method_family) == gt.method_family else 0.0


def temperature_abs_error(
    pred: RecipePrediction, gt: RecipeGroundTruth
) -> Optional[float]:
    """Absolute max-heating-temperature error in °C.

    Returns None (excluded from MAE, tracked as coverage) when either side
    is missing or the prediction is outside the physically plausible range.
    """
    if pred.error is not None:
        return None
    if pred.max_heating_temp_C is None or gt.max_heating_temp_C is None:
        return None
    if not (TEMP_MIN_C < pred.max_heating_temp_C <= TEMP_MAX_C):
        return None
    return abs(pred.max_heating_temp_C - gt.max_heating_temp_C)


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlanningResults:
    """Aggregate planning-benchmark results for one prediction method."""

    method_name: str
    n_cases: int
    n_failures: int = 0

    # Per-case raw scores (temp_abs_err entries may be None)
    per_case: dict[str, list] = field(default_factory=dict)
    predictions: list[RecipePrediction] = field(default_factory=list)

    def aggregate(self) -> dict:
        """Bootstrap-CI aggregates; MAE over scored temperature cases only."""
        out: dict = {"n_cases": self.n_cases, "n_failures": self.n_failures}

        metrics = {}
        for name in ("element_jaccard", "formula_f1", "method_accuracy"):
            scores = self.per_case.get(name, [])
            if scores:
                mean, lo, hi = bootstrap_ci(scores)
                metrics[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        out["metrics"] = metrics

        temp_errors = [e for e in self.per_case.get("temp_abs_err", []) if e is not None]
        pred_temps = [
            p.max_heating_temp_C for p in self.predictions
            if p.max_heating_temp_C is not None
        ]
        out["temperature"] = {
            "mae": float(np.mean(temp_errors)) if temp_errors else None,
            "n_scored": len(temp_errors),
            "coverage_pred": len(pred_temps) / self.n_cases if self.n_cases else 0.0,
        }

        latencies = self.per_case.get("latency", [])
        out["latency"] = {
            "mean_s": float(np.mean(latencies)) if latencies else 0.0,
            "p95_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        }
        return out

    def summary_table(self) -> str:
        """Compact ASCII summary for quick inspection."""
        agg = self.aggregate()
        lines = [
            f"Method: {self.method_name}  (n={self.n_cases}, failures={self.n_failures})",
            f"{'metric':>18}  {'mean':>8}  {'95% CI':>20}",
            "-" * 52,
        ]
        for name, m in agg["metrics"].items():
            lines.append(
                f"{name:>18}  {m['mean']:>8.4f}  "
                f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]"
            )
        t = agg["temperature"]
        mae_str = f"{t['mae']:.1f}" if t["mae"] is not None else "--"
        lines.append(
            f"{'temp MAE (°C)':>18}  {mae_str:>8}  "
            f"(scored {t['n_scored']}/{self.n_cases})"
        )
        lines.append(f"{'mean latency (s)':>18}  {agg['latency']['mean_s']:>8.2f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class PlanningBenchmark:
    """Run recipe-prediction evaluation over a held-out test set."""

    def __init__(self, test_cases: list[TestCase], verbose: bool = True):
        self.test_cases = test_cases
        self.verbose = verbose

    def evaluate(
        self, predictor: RecipePredictor, name: str = "unnamed"
    ) -> PlanningResults:
        """Evaluate *predictor* over the full test set.

        A prediction that raises is recorded as a failed RecipePrediction
        (scoring 0 on set metrics, excluded from temperature MAE); the run
        always completes.
        """
        results = PlanningResults(method_name=name, n_cases=len(self.test_cases))
        results.per_case = {
            "element_jaccard": [],
            "formula_f1": [],
            "method_accuracy": [],
            "temp_abs_err": [],
            "latency": [],
        }

        for i, case in enumerate(self.test_cases):
            if self.verbose and i % 10 == 0:
                print(f"  [{name}] {i}/{len(self.test_cases)}")

            gt = extract_ground_truth(case.raw_recipe)

            start = time.perf_counter()
            try:
                pred = predictor.predict(case)
            except Exception as e:  # noqa: BLE001 - one failure must not kill the run
                pred = RecipePrediction(
                    target_formula=case.reduced_formula, error=str(e)
                )
            if not pred.latency_s:
                pred.latency_s = time.perf_counter() - start

            if pred.error is not None:
                results.n_failures += 1

            results.predictions.append(pred)
            results.per_case["element_jaccard"].append(
                precursor_element_jaccard(pred, gt)
            )
            results.per_case["formula_f1"].append(precursor_formula_f1(pred, gt))
            results.per_case["method_accuracy"].append(method_accuracy(pred, gt))
            results.per_case["temp_abs_err"].append(temperature_abs_error(pred, gt))
            results.per_case["latency"].append(pred.latency_s)

        if self.verbose:
            print(results.summary_table())
        return results


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_planning_table(
    method_results: dict[str, PlanningResults],
    metric: str = "element_jaccard",
    alpha: float = 0.05,
) -> str:
    """LaTeX-ready comparison table with CIs and significance stars.

    Analogous to statistics.format_results_table, which is hardwired to
    retrieval per_query_sro and cannot be reused here.
    """
    from src.evaluation.statistics import wilcoxon_table

    all_scores = {
        name: res.per_case.get(metric, [])
        for name, res in method_results.items()
    }
    best_name = max(
        all_scores, key=lambda n: np.mean(all_scores[n]) if all_scores[n] else -1
    )
    pw = wilcoxon_table(all_scores, alpha=alpha)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Method & {metric.replace('_', ' ')} & 95\% CI & Effect \\",
        r"\midrule",
    ]
    for name, scores in all_scores.items():
        if not scores:
            lines.append(rf"{name} & -- & -- & -- \\")
            continue
        mean, lo, hi = bootstrap_ci(scores)
        key = (best_name, name) if (best_name, name) in pw else (name, best_name)
        sig_info = pw.get(key, {})
        sig_marker = "*" if sig_info.get("significant") and name != best_name else ""
        lines.append(
            rf"{name} & {mean:.4f}{sig_marker} & [{lo:.4f}, {hi:.4f}] & "
            rf"d={sig_info.get('effect_size', float('nan')):.2f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
