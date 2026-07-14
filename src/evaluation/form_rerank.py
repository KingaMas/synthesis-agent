"""
M1: precursor-form co-occurrence reranker.

Motivation (route-oracle analysis): the oracle gap of element-overlap
retrieval concentrates in sol-gel chemistry and is dominated by
precursor-FORM selection within a route — nitrate/alkoxide/citrate vs
oxide/carbonate, hydrates, chelators — not by route identification alone.

M1 is deliberately simple and interpretable: from the TRAIN corpus it
estimates, for each (element, route), the distribution over precursor
forms that supply that element; at query time it reranks a base
retriever's candidates by how compatible each candidate's precursor forms
are with the forms expected for the query's elements under the query's
route posterior. Counts only — no learned parameters.

Implements the BaselineRetriever protocol.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from pymatgen.core import Composition

from src.evaluation.benchmark import BaselineRetriever, _jaccard
from src.evaluation.planning import (
    KNOWN_METHODS,
    _extract_precursor_formulas,
    normalize_formula_set,
)
from src.evaluation.test_set_builder import TestCase

# Anion/ligand form classes for precursor compounds. Order matters: first
# match wins (e.g. an oxalate hydrate classifies as oxalate).
_FORM_RULES = [
    ("nitrate", ("N", "O")),
    ("carbonate", ("C", "O")),
    ("sulfate", ("S", "O")),
    ("phosphate", ("P", "O")),
    ("halide", ("Cl",)), ("halide", ("F",)), ("halide", ("Br",)), ("halide", ("I",)),
    ("hydroxide", ("H", "O")),
    ("oxide", ("O",)),
]


def precursor_form(formula: str) -> str:
    """Classify a precursor formula into a coarse anion/ligand form class."""
    try:
        comp = Composition(formula)
    except Exception:
        return "other"
    els = {str(e) for e in comp.elements}
    # organic ligands (acetates, alkoxides, citrates, oxalates): C + H (+O)
    if "C" in els and "H" in els:
        return "organic"
    for form, required in _FORM_RULES:
        if all(r in els for r in required):
            # carbonate requires C but NOT H (organic caught above)
            return form
    return "element" if len(els) == 1 else "other"


def _cation_elements(formula: str) -> set[str]:
    try:
        comp = Composition(formula)
    except Exception:
        return set()
    return {str(e) for e in comp.elements} - {"O", "H", "C", "N", "S", "Cl", "F", "Br", "I", "P"}


class FormCooccurrenceModel:
    """P(precursor form | target element, route) from train-corpus counts."""

    def __init__(self, train_corpus: list[TestCase], smoothing: float = 1.0):
        self.smoothing = smoothing
        # counts[route][element][form] and route prior per element
        self._form_counts: dict[str, dict[str, Counter]] = {
            r: defaultdict(Counter) for r in KNOWN_METHODS
        }
        self._route_counts: dict[str, Counter] = defaultdict(Counter)
        self._forms: set[str] = set()

        for tc in train_corpus:
            route = tc.synthesis_method
            if route not in KNOWN_METHODS:
                continue
            precursors = normalize_formula_set(
                _extract_precursor_formulas(tc.raw_recipe)
            )
            target_els = set(tc.elements)
            for p in precursors:
                form = precursor_form(p)
                self._forms.add(form)
                for el in _cation_elements(p) & target_els:
                    self._form_counts[route][el][form] += 1
                    self._route_counts[el][route] += 1

    def route_posterior(self, elements: list[str]) -> dict[str, float]:
        """P(route | target elements): normalised sum of per-element counts."""
        totals = Counter()
        for el in elements:
            totals.update(self._route_counts.get(el, {}))
        z = sum(totals.values())
        if z == 0:
            return {r: 1.0 / len(KNOWN_METHODS) for r in KNOWN_METHODS}
        return {r: totals.get(r, 0) / z for r in KNOWN_METHODS}

    def form_logprob(self, element: str, form: str, route: str) -> float:
        counts = self._form_counts[route].get(element, Counter())
        total = sum(counts.values())
        n_forms = max(len(self._forms), 1)
        return math.log(
            (counts.get(form, 0) + self.smoothing)
            / (total + self.smoothing * n_forms)
        )

    def _candidate_features(self, candidate: TestCase) -> list[tuple[str, frozenset]]:
        """(form, cation elements) per precursor; cached per formula."""
        if not hasattr(self, "_feat_cache"):
            self._feat_cache: dict[str, list] = {}
        key = candidate.reduced_formula
        if key not in self._feat_cache:
            feats = []
            for p in normalize_formula_set(
                _extract_precursor_formulas(candidate.raw_recipe)
            ):
                feats.append((precursor_form(p), frozenset(_cation_elements(p))))
            self._feat_cache[key] = feats
        return self._feat_cache[key]

    def compatibility(
        self, query: TestCase, candidate: TestCase, mode: str = "marginal"
    ) -> float:
        """Compatibility of the candidate's precursor forms with the query.

        mode="marginal": expected form probability under the query's route
        posterior. Diagnostic caveat: the posterior is nearly the corpus
        prior (mean P(sol-gel)=0.19 for true sol-gel train materials), so
        marginalisation effectively assumes solid-state.
        mode="max_route": score the candidate under its BEST route —
        rewards form sets coherent under SOME route, route-agnostic.
        """
        feats = self._candidate_features(candidate)
        if not feats:
            return -10.0
        query_els = set(query.elements)

        if mode == "max_route":
            route_scores = []
            for route in KNOWN_METHODS:
                scores = [
                    self.form_logprob(el, form, route)
                    for form, cation_els in feats
                    for el in cation_els & query_els
                ]
                if scores:
                    route_scores.append(sum(scores) / len(scores))
            return float(max(route_scores)) if route_scores else -10.0

        posterior = self.route_posterior(query.elements)
        scores = []
        for form, cation_els in feats:
            for el in cation_els & query_els:
                marg = sum(
                    w * math.exp(self.form_logprob(el, form, route))
                    for route, w in posterior.items()
                )
                scores.append(math.log(max(marg, 1e-12)))
        return float(sum(scores) / len(scores)) if scores else -10.0


class FormRerankRetriever:
    """Rerank a base retriever's candidates by form compatibility (M1).

    score = (1 - alpha) * element_jaccard(query, cand) + alpha * sigmoid-ish
    normalised form compatibility. alpha and fetch_factor are the only
    knobs; defaults chosen a priori (alpha=0.5, fetch 10x) — tune only on
    the train split, never on test.
    """

    def __init__(
        self,
        base: BaselineRetriever,
        model: FormCooccurrenceModel,
        alpha: float = 0.5,
        fetch_factor: int = 10,
        mode: str = "marginal",
    ):
        self.base = base
        self.model = model
        self.alpha = alpha
        self.fetch_factor = fetch_factor
        self.mode = mode

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        candidates = self.base.retrieve(query, k=k * self.fetch_factor)
        if not candidates:
            return []
        q_els = set(query.elements)
        compat = [
            self.model.compatibility(query, c, mode=self.mode) for c in candidates
        ]
        # min-max normalise compatibility to [0, 1] within the candidate set
        lo, hi = min(compat), max(compat)
        span = (hi - lo) or 1.0
        scored = [
            (
                (1 - self.alpha) * _jaccard(q_els, set(c.elements))
                + self.alpha * ((cm - lo) / span),
                c,
            )
            for cm, c in zip(compat, candidates)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]
