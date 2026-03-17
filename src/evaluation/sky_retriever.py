"""
SearchAPI adapter implementing the BaselineRetriever protocol.

SKYRetriever wraps SearchAPI to query the full Materials Project embedding
database (~100k materials), then looks up returned neighbors in the full
synthesis recipe corpus to return TestCase objects.

This correctly models the retrieval task: given a target material (possibly
a doped/non-stoichiometric composition not in the MP database), find the
most similar MP materials that have known synthesis routes.

Usage
-----
    from src.evaluation.sky_retriever import SKYRetriever
    from src.evaluation.test_set_builder import build_retrieval_corpus
    from src.search_api import SearchAPI
    from src.embedding import InputType

    corpus = build_retrieval_corpus()          # ~25k recipe materials
    api = SearchAPI(input_type=InputType.COMPOSITION, max_neighbors=200)
    retriever = SKYRetriever(api, corpus=corpus)
    benchmark.evaluate(retriever, "MAGPIE (SKY)")
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Composition

from src.evaluation.test_set_builder import TestCase


class SKYRetriever:
    """Wraps SearchAPI to implement the BaselineRetriever protocol.

    Queries the full MP embedding H5 (~100k materials) and looks up
    returned neighbors in a pre-built recipe corpus.  No refitting is
    done — the full embedding database is searched.

    Args:
        search_api:  A SearchAPI instance loaded with the composition H5.
        corpus:      All available TestCase objects (the retrieval space).
                     Pass ``build_retrieval_corpus()`` for the full recipe DB.
        fetch_factor: Multiplier on k when fetching neighbors from the H5.
                      Because most H5 materials don't have recipes, we fetch
                      ``k * fetch_factor`` candidates and return the first k
                      that have a corpus entry.
    """

    def __init__(
        self,
        search_api,
        corpus: list[TestCase],
        fetch_factor: int = 20,
    ):
        self.api = search_api
        self.fetch_factor = fetch_factor

        # Build O(1) lookup: reduced_formula → TestCase
        self._by_reduced: dict[str, TestCase] = {
            tc.reduced_formula: tc for tc in corpus
        }
        # Also index by material_id for any corpus entries that have one
        self._by_mid: dict[str, TestCase] = {
            tc.material_id: tc for tc in corpus if tc.material_id
        }

        n_h5 = len(search_api.mp_data["material_ids"])
        print(
            f"  [SKYRetriever] corpus={len(corpus)} recipe materials, "
            f"H5={n_h5} MP materials, fetch_factor={fetch_factor}"
        )

    # ------------------------------------------------------------------
    # BaselineRetriever protocol
    # ------------------------------------------------------------------

    def retrieve(self, query: TestCase, k: int) -> list[TestCase]:
        """Return up to k neighbours for *query*, excluding query itself.

        Queries the full MP H5 embedding database, then looks each
        returned neighbor up in the recipe corpus.
        """
        try:
            comp = Composition(query.reduced_formula)
        except Exception:
            return []

        n_fetch = min(k * self.fetch_factor, len(self.api.mp_data["material_ids"]))
        try:
            raw = self.api.query_with_exclusion(
                comp,
                exclude_ids=[query.material_id] if query.material_id else [],
                n_neighbors=n_fetch,
            )
        except Exception:
            return []

        results: list[TestCase] = []
        seen: set[str] = {query.reduced_formula}

        for n in raw:
            # Try material_id lookup first (faster, exact)
            tc = self._by_mid.get(n.material_id) if n.material_id else None

            # Fall back to reduced-formula lookup
            if tc is None:
                try:
                    rf = Composition(n.formula).reduced_formula
                    tc = self._by_reduced.get(rf)
                except Exception:
                    pass

            if tc is not None and tc.reduced_formula not in seen:
                seen.add(tc.reduced_formula)
                results.append(tc)
                if len(results) >= k:
                    break

        return results
