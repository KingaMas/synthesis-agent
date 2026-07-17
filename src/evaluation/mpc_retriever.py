"""
Retriever wrapper for precomputed MPC embeddings (T3, Retrieval-Retro).

The torch-side driver (scripts/rr_train_mpc.py) saves one embedding row
per material; this wrapper does cosine top-k over those rows and
implements the BaselineRetriever protocol so the MPC baseline drops
into evaluate_transferability() unchanged.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.test_set_builder import TestCase


class PrecomputedEmbeddingRetriever:
    """Cosine top-k over fixed embeddings keyed by reduced formula."""

    def __init__(
        self,
        corpus: list[TestCase],
        corpus_emb: np.ndarray,
        query_emb: dict[str, np.ndarray],
    ):
        if len(corpus) != len(corpus_emb):
            raise ValueError(
                f"corpus size {len(corpus)} != embedding rows {len(corpus_emb)}"
            )
        self.corpus = corpus
        norms = np.linalg.norm(corpus_emb, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._corpus_emb = corpus_emb / norms
        self._query_emb = query_emb

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        q = self._query_emb.get(query.reduced_formula)
        if q is None:
            return []
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return []
        sims = self._corpus_emb @ (q / qn)
        order = np.argsort(-sims, kind="stable")
        out = []
        for i in order:
            tc = self.corpus[i]
            if tc.reduced_formula == query.reduced_formula:
                continue
            out.append(tc)
            if len(out) == k:
                break
        return out
