"""
Data bridge for the Retrieval-Retro MPC baseline (T3).

Exports our TestCase corpora as plain numpy arrays that the torch-side
driver (scripts/rr_train_mpc.py, run in the .venv-rr environment) turns
into the PyG Data objects Retrieval-Retro's MPC model expects:

  comp_fea       — element-fraction vector over the fit-corpus element
                   vocabulary (their input_dim=83 analogue)
  y_lb_one       — multi-hot precursor labels over the fit-corpus
                   precursor-formula vocabulary
  y_multiple     — our corpus keeps one representative recipe per
                   formula, so this equals y_lb_one with length 1

The split into fit / early-stop-validation / query sets is decided by
the caller; this module only vectorises. Element and precursor
vocabularies always come from the TRAINING corpus alone so no test-side
information leaks into the representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymatgen.core import Composition

from src.evaluation.planning import _extract_precursor_formulas, normalize_formula_set
from src.evaluation.test_set_builder import TestCase


def _fraction_vector(formula: str, el_index: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(el_index), dtype=np.float32)
    try:
        comp = Composition(formula)
    except Exception:
        return vec
    for el, amt in comp.fractional_composition.items():
        idx = el_index.get(str(el))
        if idx is not None:
            vec[idx] = float(amt)
    return vec


def build_vocabs(train_corpus: list[TestCase]) -> tuple[list[str], list[str]]:
    """(element vocabulary, precursor-formula vocabulary) from train only."""
    elements: set[str] = set()
    precursors: set[str] = set()
    for tc in train_corpus:
        elements.update(tc.elements)
        precursors.update(
            normalize_formula_set(_extract_precursor_formulas(tc.raw_recipe))
        )
    return sorted(elements), sorted(precursors)


def export_mpc_arrays(
    train_corpus: list[TestCase],
    query_sets: dict[str, list[TestCase]],
    out_path: Path,
    train_exclude: set[str] | None = None,
) -> dict:
    """Write an .npz with composition features and precursor labels.

    train_corpus defines both vocabularies and gets labels; each named
    query set gets composition features plus labels restricted to the
    train vocabulary (queries with zero in-vocab precursors keep an
    all-zero row — the driver masks them out of any label-based metric).

    train_exclude marks corpus rows (by reduced formula) that the driver
    must not train on — used when the early-stopping query set is drawn
    from the corpus itself, so its recall measures generalisation rather
    than memorisation. Excluded rows are still embedded as corpus.
    """
    el_vocab, prec_vocab = build_vocabs(train_corpus)
    el_index = {e: i for i, e in enumerate(el_vocab)}
    prec_index = {p: i for i, p in enumerate(prec_vocab)}

    def comp_matrix(cases: list[TestCase]) -> np.ndarray:
        return np.stack([
            _fraction_vector(tc.reduced_formula, el_index) for tc in cases
        ])

    def label_matrix(cases: list[TestCase]) -> np.ndarray:
        y = np.zeros((len(cases), len(prec_vocab)), dtype=np.float32)
        for i, tc in enumerate(cases):
            for p in normalize_formula_set(
                _extract_precursor_formulas(tc.raw_recipe)
            ):
                j = prec_index.get(p)
                if j is not None:
                    y[i, j] = 1.0
        return y

    arrays = {
        "train_comp": comp_matrix(train_corpus),
        "train_y": label_matrix(train_corpus),
        "train_exclude": np.array([
            tc.reduced_formula in (train_exclude or set())
            for tc in train_corpus
        ]),
    }
    for name, cases in query_sets.items():
        arrays[f"{name}_comp"] = comp_matrix(cases)
        arrays[f"{name}_y"] = label_matrix(cases)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    meta = {
        "n_elements": len(el_vocab),
        "n_precursors": len(prec_vocab),
        "element_vocab": el_vocab,
        "sets": {
            "train": [tc.reduced_formula for tc in train_corpus],
            **{n: [tc.reduced_formula for tc in cs]
               for n, cs in query_sets.items()},
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta))
    return meta
