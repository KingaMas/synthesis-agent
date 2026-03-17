"""
Pytest configuration and shared fixtures.

All tests must pass with no API keys via:
    OFFLINE_MODE=true pytest tests/

The OFFLINE_MODE env var switches SynthesisAgent to MockSynthesisAgent,
bypassing all MPRester calls.
"""

import json
import os
from pathlib import Path

import pytest

# Force offline mode for all tests unless overridden
os.environ.setdefault("OFFLINE_MODE", "true")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture: pre-built test set (small, deterministic)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_test_cases():
    """Minimal test cases derived from fixtures — no disk I/O beyond JSON."""
    from src.evaluation.test_set_builder import TestCase

    recipes_file = FIXTURES_DIR / "test_recipes.json"
    with open(recipes_file) as fh:
        recipes: dict = json.load(fh)

    cases = []
    for formula, recipe_list in recipes.items():
        if not recipe_list:
            continue
        recipe = recipe_list[0]
        from pymatgen.core import Composition
        from src.evaluation.test_set_builder import classify_synthesis_method, _extract_precursor_elements
        try:
            comp = Composition(formula)
        except Exception:
            continue
        method_text = recipe.get("synthesis_type", "") + " " + recipe.get("paragraph_string", "")
        cases.append(
            TestCase(
                material_id=f"mock-{formula}",
                formula=formula,
                reduced_formula=comp.reduced_formula,
                elements=sorted(str(el) for el in comp.elements),
                synthesis_method=classify_synthesis_method(method_text),
                precursor_elements=_extract_precursor_elements(recipe),
                raw_recipe=recipe,
            )
        )
    return cases


@pytest.fixture(scope="session")
def mock_agent():
    """MockSynthesisAgent backed by fixture data."""
    from src.evaluation.mock_llm import MockSynthesisAgent
    return MockSynthesisAgent()


@pytest.fixture(scope="session")
def mock_search_api():
    """MockSearchAPI backed by fixture neighbors."""
    from src.evaluation.mock_llm import MockSearchAPI
    return MockSearchAPI()
