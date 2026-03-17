"""
Baseline retrievers for Table 1 comparison.

All four baselines implement the BaselineRetriever protocol defined in
benchmark.py and can be passed directly to RetrievalBenchmark.evaluate().

Baselines
---------
RandomRetriever          — Random sample. Floor for all metrics.
ElementJaccardRetriever  — Rank by element-set Jaccard similarity.
StoichiometricVectorRetriever — 100-dim element-fraction cosine similarity.
FormulaTFIDFRetriever    — TF-IDF over tokenized formula strings.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Optional

import numpy as np
from pymatgen.core import Composition, Element

from src.evaluation.test_set_builder import TestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fixed element ordering for the 100-dim stoichiometric vector
_ALL_ELEMENTS = [el.symbol for el in Element][:100]
_EL_INDEX = {sym: i for i, sym in enumerate(_ALL_ELEMENTS)}


def _composition_vector(formula: str) -> np.ndarray:
    """100-dim element-fraction vector for a formula string."""
    vec = np.zeros(100, dtype=np.float32)
    try:
        comp = Composition(formula)
        for el, amt in comp.fractional_composition.items():
            idx = _EL_INDEX.get(el.symbol)
            if idx is not None:
                vec[idx] = float(amt)
    except Exception:
        pass
    return vec


def _element_set(formula: str) -> set[str]:
    try:
        comp = Composition(formula)
        return {el.symbol for el in comp.elements}
    except Exception:
        return set()


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _tokenize_formula(formula: str) -> list[str]:
    """Split a formula string into element+number tokens."""
    return re.findall(r"[A-Z][a-z]?\d*\.?\d*", formula)


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

class RandomRetriever:
    """Return a random subset of the corpus — provides a performance floor."""

    def __init__(self, corpus: list[TestCase], seed: int = 42):
        self.corpus = corpus
        self._rng = random.Random(seed)

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        pool = [tc for tc in self.corpus if tc.reduced_formula != query.reduced_formula]
        return self._rng.sample(pool, min(k, len(pool)))


class ElementJaccardRetriever:
    """Rank corpus by Jaccard similarity of element sets."""

    def __init__(self, corpus: list[TestCase]):
        self.corpus = corpus

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        q_els = set(query.elements)
        pool = [tc for tc in self.corpus if tc.reduced_formula != query.reduced_formula]
        scored = [(tc, _jaccard(q_els, set(tc.elements))) for tc in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [tc for tc, _ in scored[:k]]


class StoichiometricVectorRetriever:
    """Rank by cosine similarity of 100-dim element-fraction vectors."""

    def __init__(self, corpus: list[TestCase]):
        self.corpus = corpus
        self._vectors: dict[str, np.ndarray] = {}
        for tc in corpus:
            self._vectors[tc.reduced_formula] = _composition_vector(tc.reduced_formula)

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        q_vec = _composition_vector(query.reduced_formula)
        q_norm = np.linalg.norm(q_vec)

        pool = [tc for tc in self.corpus if tc.reduced_formula != query.reduced_formula]
        scores = []
        for tc in pool:
            c_vec = self._vectors.get(tc.reduced_formula, _composition_vector(tc.reduced_formula))
            c_norm = np.linalg.norm(c_vec)
            if q_norm < 1e-9 or c_norm < 1e-9:
                sim = 0.0
            else:
                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            scores.append((tc, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [tc for tc, _ in scores[:k]]


class FormulaTFIDFRetriever:
    """Rank by TF-IDF cosine similarity over tokenized formula strings."""

    def __init__(self, corpus: list[TestCase]):
        self.corpus = corpus
        self._build_tfidf(corpus)

    def _build_tfidf(self, corpus: list[TestCase]) -> None:
        N = len(corpus)
        # Build document-frequency counts
        df: dict[str, int] = defaultdict(int)
        self._tokens: dict[str, list[str]] = {}
        for tc in corpus:
            tokens = set(_tokenize_formula(tc.reduced_formula))
            self._tokens[tc.reduced_formula] = list(tokens)
            for tok in tokens:
                df[tok] += 1

        # IDF
        self._idf: dict[str, float] = {
            tok: np.log((N + 1) / (cnt + 1)) + 1.0
            for tok, cnt in df.items()
        }

        # Pre-compute TF-IDF vectors (stored as dicts for sparse efficiency)
        self._vecs: dict[str, dict[str, float]] = {}
        for tc in corpus:
            tokens = _tokenize_formula(tc.reduced_formula)
            tf: dict[str, float] = defaultdict(float)
            for tok in tokens:
                tf[tok] += 1.0
            total = max(len(tokens), 1)
            vec = {tok: (cnt / total) * self._idf.get(tok, 1.0)
                   for tok, cnt in tf.items()}
            self._vecs[tc.reduced_formula] = vec

    def _cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        common_keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        return dot / (norm_a * norm_b + 1e-9)

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        q_vec = self._vecs.get(
            query.reduced_formula,
            self._build_single_vec(query.reduced_formula),
        )
        pool = [tc for tc in self.corpus if tc.reduced_formula != query.reduced_formula]
        scores = []
        for tc in pool:
            c_vec = self._vecs.get(
                tc.reduced_formula,
                self._build_single_vec(tc.reduced_formula),
            )
            scores.append((tc, self._cosine(q_vec, c_vec)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tc for tc, _ in scores[:k]]

    def _build_single_vec(self, formula: str) -> dict[str, float]:
        tokens = _tokenize_formula(formula)
        tf: dict[str, float] = defaultdict(float)
        for tok in tokens:
            tf[tok] += 1.0
        total = max(len(tokens), 1)
        return {tok: (cnt / total) * self._idf.get(tok, 1.0)
                for tok, cnt in tf.items()}


import math  # noqa: E402 — needed by FormulaTFIDFRetriever._cosine
