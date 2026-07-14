"""
Test set builder for retrieval benchmark.

Loads mp_synthesis_recipes.json.gz, groups by reduced formula,
stratifies by synthesis method to avoid over-representing common oxides,
and produces a list of TestCase objects.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pymatgen.core import Composition

from src import ASSETS_DIR


# Canonical method families used throughout the evaluation framework
METHOD_FAMILIES = {
    "solid-state": ["solid state", "solid-state", "ceramic", "calcin", "sinter"],
    "hydrothermal": ["hydrothermal", "solvothermal", "autoclave"],
    "sol-gel": ["sol-gel", "sol gel", "gelation", "xerogel", "alkoxide"],
    "combustion": ["combustion", "self-propagating", "shs", "urea"],
    "precipitation": ["precipit", "coprecip", "co-precip"],
    "other": [],
}


def classify_synthesis_method(text: str) -> str:
    """Map a free-form synthesis description to a canonical method family."""
    if not text:
        return "other"
    text_lower = text.lower()
    for family, keywords in METHOD_FAMILIES.items():
        if family == "other":
            continue
        if any(kw in text_lower for kw in keywords):
            return family
    return "other"


@dataclass
class TestCase:
    """Single evaluation unit: a held-out material with its known recipe."""

    material_id: str
    formula: str
    reduced_formula: str
    elements: list[str]
    synthesis_method: str          # canonical family
    precursor_elements: list[str]  # element symbols used as precursors
    raw_recipe: dict               # full recipe dict from the JSON file


def _extract_precursor_elements(recipe: dict) -> list[str]:
    """Pull precursor element symbols from a recipe dict."""
    elements: set[str] = set()

    # 'precursors' is a list of dicts with 'material_string' or 'formula'
    for prec in recipe.get("precursors", []):
        formula_str = prec.get("material_string") or prec.get("formula", "")
        if formula_str:
            try:
                comp = Composition(formula_str)
                elements.update(str(el) for el in comp.elements)
            except Exception:
                pass

    return sorted(elements)


def load_recipes(path: Optional[Path] = None) -> list[dict]:
    """Load all recipes from the compressed JSON file."""
    if path is None:
        path = ASSETS_DIR / "mp_synthesis_recipes.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    # The file may be a list of recipes or a dict with a top-level key
    if isinstance(data, list):
        return data
    # Fallback: try common wrapper keys
    for key in ("recipes", "data", "entries"):
        if key in data:
            return data[key]
    raise ValueError(f"Unrecognised top-level structure in {path}")


def build_retrieval_corpus(
    recipes_path: Optional[Path] = None,
) -> list[TestCase]:
    """Build the full retrieval corpus: all unique formulas from all recipes.

    Used as the search space for all retrievers.  Unlike build_test_set(),
    this applies no stratification or cap — it returns every unique
    reduced formula that pymatgen can parse.

    Returns:
        ~25k TestCase objects, one per unique reduced formula.
    """
    recipes = load_recipes(recipes_path)
    seen: dict[str, TestCase] = {}

    for recipe in recipes:
        target = recipe.get("target_formula") or recipe.get("target", {}).get(
            "material_string", ""
        )
        if not target:
            continue
        try:
            comp = Composition(target)
            reduced = comp.reduced_formula
        except Exception:
            continue

        method_text = (
            recipe.get("synthesis_type", "")
            + " "
            + recipe.get("paragraph_string", "")
        )
        method = classify_synthesis_method(method_text)
        prec_elements = _extract_precursor_elements(recipe)

        if reduced not in seen:
            seen[reduced] = TestCase(
                material_id=recipe.get("target_id", ""),
                formula=target,
                reduced_formula=reduced,
                elements=sorted(str(el) for el in comp.elements),
                synthesis_method=method,
                precursor_elements=prec_elements,
                raw_recipe=recipe,
            )

    return list(seen.values())


def build_temporal_split(
    recipes_path: Optional[Path] = None,
    years_path: Optional[Path] = None,
    cutoff_year: int = 2019,
) -> tuple[list[TestCase], list[TestCase]]:
    """Publication-year split, mirroring Retrieval-Retro's year division.

    Train: unique formulas whose EARLIEST recipe predates cutoff_year.
    Test:  unique formulas whose earliest recipe is >= cutoff_year (i.e.
    materials first reported after the cutoff — the realistic "new
    material" scenario; a formula published both before and after the
    cutoff goes to train, since its recipe was already known).

    Recipes whose DOI has no resolved year are DROPPED (not silently
    assigned); the caller sees only dated materials. Years come from the
    committed results/doi_years.json produced by scripts/fetch_doi_years.py.
    """
    import json as _json

    from src import PROJECT_ROOT

    if years_path is None:
        years_path = PROJECT_ROOT / "results" / "doi_years.json"
    years: dict = _json.loads(Path(years_path).read_text())

    earliest: dict[str, int] = {}
    representative: dict[str, TestCase] = {}
    for recipe in load_recipes(recipes_path):
        doi = recipe.get("doi") or ""
        year = years.get(doi)
        if year is None:
            continue
        target = (recipe.get("target") or {}).get("material_string", "") or ""
        if not target:
            continue
        try:
            comp = Composition(target)
            reduced = comp.reduced_formula
        except Exception:
            continue

        if reduced not in earliest or year < earliest[reduced]:
            earliest[reduced] = year
            method_text = (
                recipe.get("synthesis_type", "")
                + " "
                + recipe.get("paragraph_string", "")
            )
            representative[reduced] = TestCase(
                material_id="",
                formula=target,
                reduced_formula=reduced,
                elements=sorted(str(el) for el in comp.elements),
                synthesis_method=classify_synthesis_method(method_text),
                precursor_elements=_extract_precursor_elements(recipe),
                raw_recipe=recipe,
            )

    train = [tc for f, tc in representative.items() if earliest[f] < cutoff_year]
    test = [tc for f, tc in representative.items() if earliest[f] >= cutoff_year]
    return train, test


def load_temporal_split(
    path: Optional[Path] = None,
    recipes_path: Optional[Path] = None,
) -> tuple[list[TestCase], list[TestCase]]:
    """Load the committed temporal split (results/temporal_split_2016.json).

    Cases are re-anchored to corpus recipes by (reduced formula, DOI);
    raises if any case cannot be recovered.
    """
    import json as _json

    from src import PROJECT_ROOT

    if path is None:
        path = PROJECT_ROOT / "results" / "temporal_split_2016.json"
    spec = _json.loads(Path(path).read_text())

    by_key: dict[tuple[str, str], dict] = {}
    for recipe in load_recipes(recipes_path):
        target = (recipe.get("target") or {}).get("material_string", "") or ""
        if not target:
            continue
        try:
            reduced = Composition(target).reduced_formula
        except Exception:
            continue
        by_key.setdefault((reduced, recipe.get("doi") or ""), recipe)

    def _anchor(records: list[dict], side: str) -> list[TestCase]:
        cases, missing = [], []
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
                f"{len(missing)} {side} cases could not be re-anchored "
                f"(e.g. {missing[:3]}); do NOT silently rebuild the split."
            )
        return cases

    return _anchor(spec["train"], "train"), _anchor(spec["test"], "test")


def load_retrieval_test_set(
    path: Optional[Path] = None,
    recipes_path: Optional[Path] = None,
) -> list[TestCase]:
    """Load the committed, authoritative retrieval test set.

    Like the planning test set, this is a FILE, not a seed: the seeded
    builder's candidate pool depends on which formulas the installed
    pymatgen version parses. Cases are re-anchored to corpus recipes by
    (reduced formula, DOI); raises if any case cannot be recovered.
    """
    import json as _json

    from src import PROJECT_ROOT

    if path is None:
        path = PROJECT_ROOT / "results" / "test_set_retrieval_seed42.json"
    spec = _json.loads(Path(path).read_text())

    by_key: dict[tuple[str, str], dict] = {}
    for recipe in load_recipes(recipes_path):
        target = (recipe.get("target") or {}).get("material_string", "") or ""
        if not target:
            continue
        try:
            reduced = Composition(target).reduced_formula
        except Exception:
            continue
        by_key.setdefault((reduced, recipe.get("doi") or ""), recipe)

    cases: list[TestCase] = []
    missing: list[str] = []
    for rec in spec["cases"]:
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
            f"{len(missing)} retrieval test cases could not be re-anchored "
            f"(e.g. {missing[:3]}); corpus or formula parsing changed. Do NOT "
            f"silently rebuild the test set."
        )
    return cases


def build_test_set(
    recipes_path: Optional[Path] = None,
    max_per_method: int = 200,
    seed: int = 42,
) -> list[TestCase]:
    """Build a stratified test set from the synthesis recipe database.

    Args:
        recipes_path: Override path to mp_synthesis_recipes.json.gz.
        max_per_method: Maximum test cases per synthesis method family.
        seed: Random seed for reproducible stratification.

    Returns:
        List of TestCase objects, one per unique reduced formula,
        stratified so no single method dominates.
    """
    rng = random.Random(seed)
    recipes = load_recipes(recipes_path)

    # Deduplicate by reduced formula, keep one representative recipe each
    seen: dict[str, TestCase] = {}
    for recipe in recipes:
        target = recipe.get("target_formula") or recipe.get("target", {}).get(
            "material_string", ""
        )
        if not target:
            continue
        try:
            comp = Composition(target)
            reduced = comp.reduced_formula
        except Exception:
            continue

        method_text = (
            recipe.get("synthesis_type", "")
            + " "
            + recipe.get("paragraph_string", "")
        )
        method = classify_synthesis_method(method_text)
        prec_elements = _extract_precursor_elements(recipe)

        if reduced not in seen:
            seen[reduced] = TestCase(
                material_id=recipe.get("target_id", ""),
                formula=target,
                reduced_formula=reduced,
                elements=sorted(str(el) for el in comp.elements),
                synthesis_method=method,
                precursor_elements=prec_elements,
                raw_recipe=recipe,
            )

    # Stratify: sample up to max_per_method cases per method family
    by_method: dict[str, list[TestCase]] = {}
    for tc in seen.values():
        by_method.setdefault(tc.synthesis_method, []).append(tc)

    test_cases: list[TestCase] = []
    for method_cases in by_method.values():
        rng.shuffle(method_cases)
        test_cases.extend(method_cases[:max_per_method])

    rng.shuffle(test_cases)
    return test_cases
