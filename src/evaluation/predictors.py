"""
Recipe predictors for the synthesis planning benchmark.

Implements the methods compared in the planning benchmark:

RetrievalPredictor   Copy the top-1 retrieved neighbor's ground-truth recipe.
RulePredictor        Non-LLM baseline: per-element most-frequent precursor,
                     majority method family, per-family median temperature.
DirectLLMPredictor   Single chat-completion call, structured JSON output.
SKYAgentPredictor    Benchmark variant of the SKY agent (tool loop over
                     similarity search + leakage-aware recipe lookup).
CachedPredictor      Disk-cache wrapper making LLM runs resumable.

Heavy/optional imports (openai, agents SDK) happen inside methods so this
module imports cleanly in offline test runs.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from pymatgen.core import Composition

from src.evaluation.benchmark import BaselineRetriever
from src.evaluation.planning import (
    KNOWN_METHODS,
    RecipePrediction,
    extract_ground_truth,
    normalize_method,
)
from src.evaluation.test_set_builder import TestCase, load_recipes


# ---------------------------------------------------------------------------
# Leakage-aware recipe store
# ---------------------------------------------------------------------------

class RecipeStore:
    """Recipe lookup by reduced formula, with held-out exclusion.

    Loads mp_synthesis_recipes.json.gz ONCE and indexes every recipe under
    the reduced formulas of its targets. Formulas in *exclude_formulas*
    are never indexed, so a lookup can never return a held-out recipe.
    """

    def __init__(
        self,
        recipes_path: Optional[Path] = None,
        exclude_formulas: frozenset[str] = frozenset(),
    ):
        self.exclude_formulas = set(exclude_formulas)
        self._index: dict[str, list[dict]] = defaultdict(list)

        for recipe in load_recipes(recipes_path):
            targets = recipe.get("targets_formula_s") or []
            if not targets:
                target = (recipe.get("target") or {}).get("material_string", "")
                targets = [target] if target else []
            for target in targets:
                try:
                    reduced = Composition(target).reduced_formula
                except Exception:
                    continue
                if reduced in self.exclude_formulas:
                    continue
                self._index[reduced].append(recipe)

    def lookup(self, formula: str) -> list[dict]:
        """Recipes whose target matches *formula* (reduced-form comparison)."""
        try:
            reduced = Composition(formula).reduced_formula
        except Exception:
            return []
        return self._index.get(reduced, [])

    def __len__(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Structured-output parsing (shared by both LLM predictors)
# ---------------------------------------------------------------------------

PREDICTION_JSON_SCHEMA = (
    '{"precursors": ["<formula>", ...], '
    '"method": "solid-state" | "sol-gel", '
    '"max_heating_temperature_C": <number or null>, '
    '"rationale": "<1-2 sentences>"}'
)


def parse_prediction_json(text: str, target_formula: str) -> RecipePrediction:
    """Parse an LLM response into a RecipePrediction.

    Tolerates markdown fences and JSON embedded in prose. On failure
    returns a prediction with error="parse" but raw_text preserved, so
    cached responses can be re-scored after parser improvements.
    """
    raw = text or ""
    candidate = raw.strip()
    # Strip markdown fences
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.MULTILINE)

    data = None
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        # Fall back to brace-delimited blocks embedded in prose. Try the
        # widest span first, then narrower spans anchored at each later
        # opening brace (an agent transcript's final answer is usually the
        # last block).
        end = candidate.rfind("}")
        if end != -1:
            starts = [m.start() for m in re.finditer(r"\{", candidate[: end + 1])]
            for start in starts:
                try:
                    data = json.loads(candidate[start : end + 1])
                    break
                except (json.JSONDecodeError, ValueError):
                    continue

    if not isinstance(data, dict):
        return RecipePrediction(
            target_formula=target_formula, raw_text=raw, error="parse"
        )

    precursors = data.get("precursors") or []
    if not isinstance(precursors, list):
        precursors = [str(precursors)]
    precursors = [str(p) for p in precursors if p]

    temp = data.get("max_heating_temperature_C")
    try:
        temp = float(temp) if temp is not None else None
    except (TypeError, ValueError):
        temp = None

    return RecipePrediction(
        target_formula=target_formula,
        precursor_formulas=precursors,
        method_family=normalize_method(str(data.get("method", ""))),
        max_heating_temp_C=temp,
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# Retrieval-only predictor
# ---------------------------------------------------------------------------

class RetrievalPredictor:
    """Predict by copying the top-1 neighbor's ground-truth recipe.

    The retriever's corpus must already have held-out formulas removed
    (see planning.split_heldout).
    """

    def __init__(self, retriever: BaselineRetriever, name: str = "retrieval"):
        self.retriever = retriever
        self.name = name

    def predict(self, case: TestCase) -> RecipePrediction:
        neighbors = self.retriever.retrieve(case, k=1)
        if not neighbors:
            return RecipePrediction(
                target_formula=case.reduced_formula, error="no neighbor"
            )
        neighbor = neighbors[0]
        gt = extract_ground_truth(neighbor.raw_recipe)
        return RecipePrediction(
            target_formula=case.reduced_formula,
            precursor_formulas=gt.precursor_formulas,
            method_family=gt.method_family,
            max_heating_temp_C=gt.max_heating_temp_C,
            provenance=neighbor.reduced_formula,
        )


# ---------------------------------------------------------------------------
# Rule-based predictor
# ---------------------------------------------------------------------------

# Common commercial precursors, used when an element was never seen in training
FALLBACK_PRECURSORS = {
    "Li": "Li2CO3", "Na": "Na2CO3", "K": "K2CO3",
    "Mg": "MgO", "Ca": "CaCO3", "Sr": "SrCO3", "Ba": "BaCO3",
    "Ti": "TiO2", "Zr": "ZrO2", "V": "V2O5", "Nb": "Nb2O5", "Ta": "Ta2O5",
    "Cr": "Cr2O3", "Mo": "MoO3", "W": "WO3", "Mn": "MnO2",
    "Fe": "Fe2O3", "Co": "Co3O4", "Ni": "NiO", "Cu": "CuO", "Zn": "ZnO",
    "Al": "Al2O3", "Ga": "Ga2O3", "In": "In2O3", "Si": "SiO2",
    "Sn": "SnO2", "Pb": "PbO", "Bi": "Bi2O3", "La": "La2O3",
    "Ce": "CeO2", "Y": "Y2O3", "P": "NH4H2PO4", "B": "H3BO3",
}

# Elements supplied by the atmosphere or precursor anions, not a dedicated
# precursor compound
NON_PRECURSOR_ELEMENTS = {"O", "H", "C", "N"}


class RulePredictor:
    """Deterministic non-LLM baseline fitted on the training corpus.

    Precursors: for each target element, the most frequent training-corpus
    precursor containing that element (fallback table for unseen elements).
    Method: majority family in training corpus. Temperature: per-family
    median of ground-truth max heating temperatures.
    """

    name = "rule"

    def __init__(self, train_corpus: list[TestCase]):
        precursor_counts: dict[str, Counter] = defaultdict(Counter)
        method_counts: Counter = Counter()
        temps_by_method: dict[str, list[float]] = defaultdict(list)

        for tc in train_corpus:
            gt = extract_ground_truth(tc.raw_recipe)
            method_counts[gt.method_family] += 1
            if gt.max_heating_temp_C is not None:
                temps_by_method[gt.method_family].append(gt.max_heating_temp_C)
            for formula in gt.precursor_formulas:
                try:
                    comp = Composition(formula)
                except Exception:
                    continue
                reduced = comp.reduced_formula
                for el in comp.elements:
                    precursor_counts[str(el)][reduced] += 1

        self._precursor_for_element = {
            el: counts.most_common(1)[0][0]
            for el, counts in precursor_counts.items()
        }
        self._majority_method = (
            method_counts.most_common(1)[0][0] if method_counts else "solid-state"
        )
        self._median_temp = {
            method: float(np.median(temps))
            for method, temps in temps_by_method.items()
        }

    def predict(self, case: TestCase) -> RecipePrediction:
        precursors: list[str] = []
        for el in case.elements:
            if el in NON_PRECURSOR_ELEMENTS and el not in FALLBACK_PRECURSORS:
                continue
            if el == "O":  # oxygen comes from oxide precursors / atmosphere
                continue
            chosen = self._precursor_for_element.get(el) or FALLBACK_PRECURSORS.get(el)
            if chosen:
                precursors.append(chosen)
        return RecipePrediction(
            target_formula=case.reduced_formula,
            precursor_formulas=sorted(set(precursors)),
            method_family=self._majority_method,
            max_heating_temp_C=self._median_temp.get(self._majority_method),
        )


# ---------------------------------------------------------------------------
# Direct LLM predictor (no tools)
# ---------------------------------------------------------------------------

DIRECT_LLM_SYSTEM_PROMPT = (
    "You are an expert solid-state chemist. Propose a realistic laboratory "
    "synthesis recipe for the requested target material."
)

DIRECT_LLM_USER_TEMPLATE = """Target material: {formula}

Propose the most likely synthesis recipe. Return ONLY a JSON object:
{schema}

precursors: commercially available starting compounds (e.g. "BaCO3", "TiO2").
method: choose exactly one of "solid-state" or "sol-gel".
max_heating_temperature_C: the highest calcination/sintering temperature in Celsius, or null."""


class DirectLLMPredictor:
    """Single chat-completion call with structured JSON output.

    Fills the gap left by the empty SynthesisLLMAgent stub in src/agent.py;
    lives here to avoid that module's MP_API_KEY import-time requirement.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.name = f"llm[{model}]"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os

            import openai

            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MDG_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY or OPENAI_MDG_API_KEY not set")
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def predict(self, case: TestCase) -> RecipePrediction:
        client = self._get_client()
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": DIRECT_LLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DIRECT_LLM_USER_TEMPLATE.format(
                        formula=case.reduced_formula, schema=PREDICTION_JSON_SCHEMA
                    ),
                },
            ],
        )
        latency = time.perf_counter() - start
        pred = parse_prediction_json(
            response.choices[0].message.content, case.reduced_formula
        )
        pred.latency_s = latency
        pred.model = self.model
        return pred


# ---------------------------------------------------------------------------
# Cache wrapper
# ---------------------------------------------------------------------------

class CachedPredictor:
    """Disk-cache wrapper: one JSON file per query formula.

    Makes LLM runs resumable and re-scoring free. Failed predictions are
    cached too; pass retry_failures=True to re-attempt them. API errors
    are retried with exponential backoff before being recorded as failures.
    """

    def __init__(
        self,
        inner,
        cache_dir: Path,
        retry_failures: bool = False,
        max_retries: int = 3,
    ):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir) / _slug(inner.name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retry_failures = retry_failures
        self.max_retries = max_retries

    def _cache_path(self, case: TestCase) -> Path:
        return self.cache_dir / f"{_slug(case.reduced_formula)}.json"

    def predict(self, case: TestCase) -> RecipePrediction:
        path = self._cache_path(case)
        if path.exists():
            cached = RecipePrediction(**json.loads(path.read_text()))
            if cached.error is None or not self.retry_failures:
                return cached

        pred = self._predict_with_retries(case)
        path.write_text(json.dumps(pred.__dict__, default=str, indent=1))
        return pred

    def _predict_with_retries(self, case: TestCase) -> RecipePrediction:
        delay = 2.0
        last_error = "unknown"
        for attempt in range(self.max_retries):
            try:
                return self.inner.predict(case)
            except Exception as e:  # noqa: BLE001 - record and retry API errors
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
        return RecipePrediction(
            target_formula=case.reduced_formula, error=last_error
        )


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text)
