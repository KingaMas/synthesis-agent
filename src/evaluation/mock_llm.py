"""
MockSynthesisAgent — offline-safe drop-in for SynthesisAgent.

Returns pre-cached recipe data from tests/fixtures/ without making any
API calls.  Set OFFLINE_MODE=true to force all tests to use this class.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pymatgen.core import Composition, Structure

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


class MockSearchAPI:
    """Returns pre-cached KNN neighbors from a JSON fixture file."""

    def __init__(self, fixture_file: Optional[Path] = None):
        if fixture_file is None:
            fixture_file = FIXTURES_DIR / "test_neighbors.json"
        with open(fixture_file) as fh:
            self._data: dict = json.load(fh)

    def query(self, input_data: Composition | Structure, n_neighbors: int = 10):
        from src.schema import Neighbor
        formula = (
            input_data.reduced_formula
            if isinstance(input_data, Composition)
            else str(input_data.composition.reduced_formula)
        )
        raw = self._data.get(formula, self._data.get("default", []))
        results = []
        for i, entry in enumerate(raw[:n_neighbors]):
            results.append(
                Neighbor(
                    neighbor_index=i,
                    material_id=entry.get("material_id", f"mock-{i}"),
                    formula=entry.get("formula", "Fe2O3"),
                    distance=float(entry.get("distance", 1.0)),
                    confidence=float(entry.get("confidence", 0.5)),
                )
            )
        return results

    def query_with_exclusion(
        self,
        input_data: Composition | Structure,
        exclude_ids: list[str],
        n_neighbors: int = 10,
    ):
        all_results = self.query(input_data, n_neighbors=n_neighbors + len(exclude_ids) + 5)
        exclude_set = set(exclude_ids)
        filtered = [r for r in all_results if r.material_id not in exclude_set]
        return filtered[:n_neighbors]


class MockSynthesisAgent:
    """Offline SynthesisAgent that uses fixture data instead of the MP API.

    Usage:
        import os; os.environ["OFFLINE_MODE"] = "true"
        from src.evaluation.mock_llm import MockSynthesisAgent
        agent = MockSynthesisAgent()
    """

    def __init__(
        self,
        neighbors_fixture: Optional[Path] = None,
        recipes_fixture: Optional[Path] = None,
    ):
        self.search_api_composition = MockSearchAPI(neighbors_fixture)
        self.search_api_structure = MockSearchAPI(neighbors_fixture)

        recipes_file = recipes_fixture or (FIXTURES_DIR / "test_recipes.json")
        with open(recipes_file) as fh:
            self._recipes: dict = json.load(fh)

    def find_similar_materials_by_composition(
        self, composition_str: str, n_neighbors: int = 10
    ):
        comp = Composition(composition_str)
        return self.search_api_composition.query(comp, n_neighbors=n_neighbors)

    def find_similar_materials_by_structure(self, structure, n_neighbors: int = 10):
        return self.search_api_structure.query(structure, n_neighbors=n_neighbors)

    def get_synthesis_recipes_by_formula(self, formula: str) -> list:
        """Return cached recipes for formula, empty list if not found."""
        try:
            reduced = Composition(formula).reduced_formula
        except Exception:
            reduced = formula
        return self._recipes.get(reduced, [])

    def get_summarydoc_by_material_id(self, material_id: str) -> list:
        return []

    def get_structure_by_material_id(self, material_id: str):
        return None


def get_agent(offline: Optional[bool] = None):
    """Return real or mock agent depending on OFFLINE_MODE env var.

    Args:
        offline: Override env var if provided.

    Returns:
        SynthesisAgent (real) or MockSynthesisAgent (offline).
    """
    if offline is None:
        offline = os.getenv("OFFLINE_MODE", "").lower() in {"1", "true", "yes"}
    if offline:
        return MockSynthesisAgent()
    from src.agent import SynthesisAgent
    return SynthesisAgent()
