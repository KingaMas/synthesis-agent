#!/usr/bin/env python
"""
T5: reconcile our temperature MAE with Prein et al.'s protocol.

Three analyses, written to results/temperature_reconciliation.json:

1. coverage      — how many solid-state recipes report a calcination /
                   sintering / any-heating temperature at all;
2. label noise   — leave-one-out replication MAE across recipes of the
                   SAME formula from different papers (solid-state),
                   per protocol: the floor imposed by the ground truth;
3. protocol eval — rule (train median) and element-Jaccard
                   retrieval-copy MAEs under max-heating vs
                   per-operation protocols, on the ladder validation
                   solid-state queries (fit <2014 corpus, 2014-15
                   queries; the frozen test split is not touched).

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/run_temperature_reconciliation.py
"""

import json
from collections import defaultdict
from pathlib import Path


def main():
    from pymatgen.core import Composition

    from src.evaluation.baselines import ElementJaccardRetriever
    from src.evaluation.planning import extract_max_heating_temp_C
    from src.evaluation.temperature import (
        PROTOCOLS,
        ProtocolRetrievalCopyPredictor,
        ProtocolRulePredictor,
        evaluate_protocols,
        protocol_temp_C,
        replication_noise,
    )
    from src.evaluation.test_set_builder import (
        classify_synthesis_method,
        load_ladder_validation_split,
        load_recipes,
    )

    results: dict = {"_config": {
        "protocols": list(PROTOCOLS),
        "route_scope": "solid-state only (Prein et al. protocol)",
        "eval_split": "ladder validation (fit <2014, queries 2014-15); "
                      "frozen 2016 test split not consumed",
    }}

    # ---- 1+2: corpus-wide coverage and replication noise -------------
    print("Grouping solid-state recipes by formula ...")
    by_formula: dict[str, list[dict]] = defaultdict(list)
    coverage = {p: 0 for p in PROTOCOLS}
    n_solid = n_corrupted = 0
    for recipe in load_recipes():
        method_text = (recipe.get("synthesis_type", "") + " "
                       + recipe.get("paragraph_string", ""))
        if classify_synthesis_method(method_text) != "solid-state":
            continue
        target = (recipe.get("target") or {}).get("material_string", "") or ""
        if not target:
            continue
        try:
            reduced = Composition(target).reduced_formula
        except Exception:
            continue
        n_solid += 1
        by_formula[reduced].append(recipe)
        for p in PROTOCOLS:
            if protocol_temp_C(recipe, p) is not None:
                coverage[p] += 1
        raw = extract_max_heating_temp_C(recipe)
        guarded = protocol_temp_C(recipe, "max_heating")
        if raw is not None and raw != guarded:
            n_corrupted += 1

    results["coverage"] = {
        "n_solid_state_recipes": n_solid,
        "n_max_heating_gt_corrupted_by_implausible_values": n_corrupted,
        **{p: {"n_with_temp": coverage[p],
               "fraction": round(coverage[p] / n_solid, 4)}
           for p in PROTOCOLS},
    }
    print(f"  {n_solid} solid-state recipes; coverage: "
          + ", ".join(f"{p} {coverage[p]}" for p in PROTOCOLS))

    print("Computing replication-noise floor ...")
    results["replication_noise"] = {
        p: replication_noise(by_formula, p) for p in PROTOCOLS
    }
    for p, r in results["replication_noise"].items():
        print(f"  {p}: LOO-MAE {r['loo_mae']:.1f} degC over "
              f"{r['n_reports']} reports / {r['n_formulas']} formulas")

    # ---- 3: protocol evaluation on the ladder validation split -------
    print("Loading ladder validation split ...")
    fit, val = load_ladder_validation_split()
    fit_solid = [tc for tc in fit if tc.synthesis_method == "solid-state"]
    val_solid = [tc for tc in val if tc.synthesis_method == "solid-state"]
    print(f"  fit solid-state {len(fit_solid)}, "
          f"val solid-state queries {len(val_solid)}")

    predictors = {
        "rule_median": ProtocolRulePredictor(fit_solid),
        "retrieval_copy_jaccard": ProtocolRetrievalCopyPredictor(
            ElementJaccardRetriever(corpus=fit_solid)
        ),
    }
    results["protocol_eval"] = evaluate_protocols(predictors, val_solid)
    results["_config"]["n_fit_solid"] = len(fit_solid)
    results["_config"]["n_val_solid"] = len(val_solid)

    out = Path("results/temperature_reconciliation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\n{'=' * 68}")
    print(f"{'protocol':14s} {'noise floor':>12s} "
          f"{'rule MAE':>10s} {'retr MAE':>10s} {'n':>5s}")
    for p in PROTOCOLS:
        noise = results["replication_noise"][p]["loo_mae"]
        ev = results["protocol_eval"]["per_protocol_support"][p]
        print(f"{p:14s} {noise:12.1f} "
              f"{ev['rule_median']['mae']:10.1f} "
              f"{ev['retrieval_copy_jaccard']['mae']:10.1f} {ev['n']:5d}")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
