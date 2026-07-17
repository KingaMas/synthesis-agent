"""
M2: gradient-boosted pairwise ranker (LambdaMART) over hand features.

M1 (form_rerank.py) failed its validation criterion because its fixed
count-marginalisation collapses to the corpus route prior. M2 asks whether
a learned ranker can exploit the same signals — plus composition
similarity, Magpie distance, and electronegativity/valence deltas — when
it is free to weight them per query. Features and labels follow the
handoff spec: hand features per (query, candidate) pair, recipe-overlap
labels (formula-set Jaccard of normalised precursor sets), temporal
hygiene throughout (featurizer statistics, co-occurrence counts and
training pairs all come from the fit corpus only).

Implements the BaselineRetriever protocol via PairwiseRankRetriever.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Composition, Element

from src.evaluation.baselines import _composition_vector
from src.evaluation.benchmark import BaselineRetriever, _jaccard
from src.evaluation.form_rerank import (
    FormCooccurrenceModel,
    _cation_elements,
    precursor_form,
)
from src.evaluation.planning import KNOWN_METHODS
from src.evaluation.test_set_builder import TestCase
from src.evaluation.transferability import FormulaSetIndex

# Candidate precursor form classes (superset of form_rerank's rule outputs)
_FORM_CLASSES = (
    "nitrate", "carbonate", "sulfate", "phosphate", "halide",
    "hydroxide", "oxide", "organic", "element", "other",
)

# Non-metal elements used to compare target anion chemistry
_ANION_ELS = frozenset({"O", "H", "C", "N", "S", "P", "Se", "Te",
                        "Cl", "F", "Br", "I"})


def _elem_props(symbol: str) -> tuple[float, float]:
    """(Pauling electronegativity, mean common oxidation state magnitude).

    Dummy species in text-mined formulas (e.g. the 'A' in 'A(FeO2)2')
    are not Elements; they contribute NaN and are skipped upstream.
    """
    try:
        el = Element(symbol)
    except ValueError:
        return np.nan, np.nan
    x = el.X if el.X is not None else np.nan
    oxi = el.common_oxidation_states
    val = float(np.mean([abs(o) for o in oxi])) if oxi else np.nan
    return float(x), val


class PairFeatureExtractor:
    """Per-(query, candidate) hand features, fit on the train corpus only.

    Feature groups (names in .feature_names):
      similarity   — element Jaccard, shared-element count, size deltas,
                     stoichiometric cosine
      magpie       — Euclidean and cosine distance between standardised
                     Magpie composition vectors (statistics from fit corpus)
      chemistry    — electronegativity / valence deltas, anion-set overlap
      route        — query route posterior, candidate route one-hot, and
                     the posterior mass on the candidate's route
      forms        — candidate precursor form-class indicators
    """

    def __init__(self, train_corpus: list[TestCase]):
        self._cooc = FormCooccurrenceModel(train_corpus)
        self._index = FormulaSetIndex(train_corpus)

        from matminer.featurizers.composition import ElementProperty
        self._magpie = ElementProperty.from_preset("magpie")
        self._magpie_cache: dict[str, np.ndarray] = {}

        mats = np.array([self._magpie_vec(tc.reduced_formula)
                         for tc in train_corpus])
        self._mag_mean = np.nanmean(mats, axis=0)
        self._mag_std = np.nanstd(mats, axis=0)
        self._mag_std[self._mag_std < 1e-9] = 1.0

        self._prop_cache: dict[str, tuple[float, float, float, frozenset]] = {}
        self._form_cache: dict[str, np.ndarray] = {}
        self._posterior_cache: dict[str, dict[str, float]] = {}
        self._stoich_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._std_magpie_cache: dict[str, tuple[np.ndarray, float]] = {}

        self.feature_names = (
            ["el_jaccard", "n_shared_els", "n_els_q", "n_els_c",
             "n_els_absdiff", "stoich_cosine",
             "magpie_dist", "magpie_cosine",
             "mean_X_delta", "max_X_delta", "mean_val_delta",
             "anion_jaccard", "both_oxide_target"]
            + [f"q_post_{r}" for r in KNOWN_METHODS]
            + ["post_on_cand_route"]
            + [f"cand_route_{r}" for r in KNOWN_METHODS]
            + [f"cand_form_{f}" for f in _FORM_CLASSES]
        )

    # -- cached per-formula pieces -------------------------------------
    def _magpie_vec(self, formula: str) -> np.ndarray:
        v = self._magpie_cache.get(formula)
        if v is None:
            try:
                v = np.asarray(
                    self._magpie.featurize(Composition(formula)), dtype=float
                )
            except Exception:
                v = np.full(len(self._magpie.feature_labels()), np.nan)
            self._magpie_cache[formula] = v
        return v

    def _std_magpie(self, formula: str) -> tuple[np.ndarray, float]:
        got = self._std_magpie_cache.get(formula)
        if got is None:
            v = (self._magpie_vec(formula) - self._mag_mean) / self._mag_std
            v = np.nan_to_num(v, nan=0.0)
            got = (v, float(np.linalg.norm(v)))
            self._std_magpie_cache[formula] = got
        return got

    def _stoich(self, formula: str) -> tuple[np.ndarray, float]:
        got = self._stoich_cache.get(formula)
        if got is None:
            v = _composition_vector(formula)
            got = (v, float(np.linalg.norm(v)))
            self._stoich_cache[formula] = got
        return got

    def _props(self, tc: TestCase):
        """(mean X, max X, mean |oxidation state|, anion element set)."""
        got = self._prop_cache.get(tc.reduced_formula)
        if got is None:
            xs, vals = [], []
            for el in tc.elements:
                x, v = _elem_props(el)
                if np.isfinite(x):
                    xs.append(x)
                if np.isfinite(v):
                    vals.append(v)
            got = (
                float(np.mean(xs)) if xs else 0.0,
                float(np.max(xs)) if xs else 0.0,
                float(np.mean(vals)) if vals else 0.0,
                frozenset(tc.elements) & _ANION_ELS,
            )
            self._prop_cache[tc.reduced_formula] = got
        return got

    def _cand_forms(self, tc: TestCase) -> np.ndarray:
        v = self._form_cache.get(tc.reduced_formula)
        if v is None:
            forms = {precursor_form(p) for p in self._index.get(tc)}
            v = np.array([1.0 if f in forms else 0.0 for f in _FORM_CLASSES])
            self._form_cache[tc.reduced_formula] = v
        return v

    def _posterior(self, tc: TestCase) -> dict[str, float]:
        p = self._posterior_cache.get(tc.reduced_formula)
        if p is None:
            p = self._cooc.route_posterior(tc.elements)
            self._posterior_cache[tc.reduced_formula] = p
        return p

    # -- the feature vector --------------------------------------------
    def features(self, query: TestCase, candidate: TestCase) -> np.ndarray:
        q_els, c_els = set(query.elements), set(candidate.elements)

        qv, qn = self._stoich(query.reduced_formula)
        cv, cn = self._stoich(candidate.reduced_formula)
        stoich_cos = float(qv @ cv / (qn * cn)) if qn > 1e-9 and cn > 1e-9 else 0.0

        qm, qmn = self._std_magpie(query.reduced_formula)
        cm, cmn = self._std_magpie(candidate.reduced_formula)
        mag_dist = float(np.linalg.norm(qm - cm)) / len(qm) ** 0.5
        mag_cos = float(qm @ cm / (qmn * cmn)) if qmn > 1e-9 and cmn > 1e-9 else 0.0

        q_meanX, q_maxX, q_val, q_anions = self._props(query)
        c_meanX, c_maxX, c_val, c_anions = self._props(candidate)

        post = self._posterior(query)
        cand_route = candidate.synthesis_method

        row = [
            _jaccard(q_els, c_els),
            float(len(q_els & c_els)),
            float(len(q_els)), float(len(c_els)),
            float(abs(len(q_els) - len(c_els))),
            stoich_cos,
            mag_dist, mag_cos,
            abs(q_meanX - c_meanX), abs(q_maxX - c_maxX),
            abs(q_val - c_val),
            _jaccard(q_anions, c_anions),
            1.0 if ("O" in q_els and "O" in c_els) else 0.0,
        ]
        row += [post.get(r, 0.0) for r in KNOWN_METHODS]
        row.append(post.get(cand_route, 0.0))
        row += [1.0 if cand_route == r else 0.0 for r in KNOWN_METHODS]
        row += list(self._cand_forms(candidate))
        return np.asarray(row, dtype=float)


class GradientBoostedRanker:
    """LambdaMART over PairFeatureExtractor features, recipe-overlap labels.

    Training groups are queries drawn from the fit corpus; candidates per
    group come from the base retriever (leave-one-out within the corpus),
    so the ranker learns to reorder exactly the kind of candidate list it
    will see at retrieval time. Labels are formula-set Jaccard overlaps
    discretised to 0-10 for the lambdarank objective.
    """

    def __init__(
        self,
        extractor: PairFeatureExtractor,
        n_candidates: int = 50,
        num_leaves: int = 31,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        min_child_samples: int = 40,
        seed: int = 42,
    ):
        self.extractor = extractor
        self.n_candidates = n_candidates
        self.params = dict(
            objective="lambdarank",
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            random_state=seed,
            verbose=-1,
        )
        self.model = None

    def fit(
        self,
        train_corpus: list[TestCase],
        base: BaselineRetriever,
        verbose: bool = True,
    ) -> "GradientBoostedRanker":
        import lightgbm as lgb

        index = self.extractor._index
        X, y, groups = [], [], []
        for i, q in enumerate(train_corpus):
            if verbose and i % 500 == 0:
                print(f"  building training pairs {i}/{len(train_corpus)}")
            cands = base.retrieve(q, k=self.n_candidates)
            if len(cands) < 2:
                continue
            q_set = set(index.get(q))
            labels = [
                int(round(10 * _jaccard(q_set, set(index.get(c)))))
                for c in cands
            ]
            if len(set(labels)) < 2:      # no signal to rank in this group
                continue
            X.extend(self.extractor.features(q, c) for c in cands)
            y.extend(labels)
            groups.append(len(cands))

        X = np.asarray(X)
        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X, np.asarray(y), group=groups)
        if verbose:
            print(f"  fitted on {len(groups)} groups, {len(X)} pairs")
        return self

    def feature_importances(self) -> dict[str, float]:
        imp = self.model.feature_importances_
        return dict(sorted(
            zip(self.extractor.feature_names, imp.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ))


class PairwiseRankRetriever:
    """Rerank a base retriever's candidate list with the M2 ranker."""

    def __init__(
        self,
        base: BaselineRetriever,
        ranker: GradientBoostedRanker,
        fetch_factor: int = 10,
    ):
        self.base = base
        self.ranker = ranker
        self.fetch_factor = fetch_factor

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        candidates = self.base.retrieve(query, k=k * self.fetch_factor)
        if not candidates:
            return []
        X = np.asarray(
            [self.ranker.extractor.features(query, c) for c in candidates]
        )
        scores = self.ranker.model.predict(X)
        order = np.argsort(-scores, kind="stable")
        return [candidates[i] for i in order[:k]]
