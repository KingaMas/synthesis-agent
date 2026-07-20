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


class ScoreMatrixRetriever:
    """Top-k over a precomputed (query x corpus) score matrix.

    Used for the Retrieval-Retro NRE baseline, whose score is a Gibbs
    proxy computed offline in the torch venv. smallest=True ranks
    ascending (their delta-G convention: smaller is better).
    """

    def __init__(
        self,
        corpus: list[TestCase],
        scores: dict[str, np.ndarray],
        smallest: bool = True,
    ):
        self.corpus = corpus
        self._scores = scores
        self._sign = 1.0 if smallest else -1.0

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        row = self._scores.get(query.reduced_formula)
        if row is None:
            return []
        order = np.argsort(self._sign * row, kind="stable")
        out = []
        for i in order:
            tc = self.corpus[i]
            if tc.reduced_formula == query.reduced_formula:
                continue
            out.append(tc)
            if len(out) == k:
                break
        return out


class UnionRetriever:
    """Their reference-set fusion: take the first retrievers' top picks
    in order, skipping duplicates, until k candidates are collected.
    Quotas follow Retrieval-Retro's usage (K=3 from MPC, rest from NRE).
    """

    def __init__(self, retrievers: list, quotas: list[int]):
        assert len(retrievers) == len(quotas)
        self.retrievers = retrievers
        self.quotas = quotas

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        out: list[TestCase] = []
        seen: set[str] = set()

        def take(retriever, quota: int) -> None:
            for tc in retriever.retrieve(query, k=k):
                if quota <= 0 or len(out) == k:
                    return
                if tc.reduced_formula in seen:
                    continue
                seen.add(tc.reduced_formula)
                out.append(tc)
                quota -= 1

        for retriever, quota in zip(self.retrievers, self.quotas):
            take(retriever, quota)
        # top up from the first retriever if a component undershot k
        if len(out) < k:
            take(self.retrievers[0], k - len(out))
        return out


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
