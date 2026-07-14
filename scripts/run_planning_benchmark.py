#!/usr/bin/env python
"""
Run the synthesis planning benchmark (Table 2).

Evaluates recipe-prediction methods against a held-out set of ground-truth
recipes sampled from assets/mp_synthesis_recipes.json.gz. Held-out formulas
are excluded from every retrieval corpus and recipe store (leakage guard).

Methods
-------
rule                Non-LLM baseline (element→precursor rules, majority method)
retrieval-jaccard   Copy top-1 Element-Jaccard neighbor's recipe
retrieval-sky       Copy top-1 MAGPIE-embedding neighbor's recipe
llm                 Direct LLM prompting, no tools (needs OPENAI_API_KEY)
sky                 Full SKY agent with tool loop (needs OPENAI_API_KEY)

Usage
-----
    python scripts/run_planning_benchmark.py --methods rule retrieval-jaccard
    python scripts/run_planning_benchmark.py --max-cases 5           # LLM smoke test
    python scripts/run_planning_benchmark.py                         # full run

Environment
-----------
rule / retrieval-* need only local assets. llm / sky need OPENAI_API_KEY
(or OPENAI_MDG_API_KEY).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pick up OPENAI_API_KEY / MP_API_KEY from .env

# Run from repo root: PYTHONPATH=. .venv/bin/python3 scripts/run_planning_benchmark.py

ALL_METHODS = ["rule", "retrieval-jaccard", "retrieval-sky", "llm", "sky"]


def main():
    parser = argparse.ArgumentParser(
        description="Run SKY synthesis planning benchmark (Table 2)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
        help="Prediction methods to evaluate (default: all)",
    )
    parser.add_argument("--sky-model", default="gpt-4o-mini",
                        help="Model for the SKY agent method (default: gpt-4o-mini)")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="Model for direct LLM prompting (default: gpt-4o-mini)")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Truncate test set for smoke runs")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("results/planning_cache"),
                        help="LLM response cache directory")
    parser.add_argument("--output", type=Path,
                        default=Path("results/planning_results.json"))
    parser.add_argument("--rubric", action="store_true",
                        help="Also score raw LLM outputs with RubricEvaluator")
    parser.add_argument("--retry-failures", action="store_true",
                        help="Re-attempt cached failed predictions")
    args = parser.parse_args()

    from src.evaluation.planning import (
        PlanningBenchmark,
        format_planning_table,
        load_planning_test_set,
        split_heldout,
    )
    from src.evaluation.predictors import (
        CachedPredictor,
        RetrievalPredictor,
        RulePredictor,
    )
    from src.evaluation.statistics import holm_correction, wilcoxon_table
    from src.evaluation.test_set_builder import build_retrieval_corpus

    # ------------------------------------------------------------------ #
    # Held-out test set + leakage-filtered training corpus
    # ------------------------------------------------------------------ #
    # The test set is a committed FILE, not a seed: the seeded builder's
    # candidate pool depends on the installed pymatgen version.
    print("Loading committed held-out planning test set ...")
    test_cases = load_planning_test_set()
    if args.max_cases:
        test_cases = test_cases[: args.max_cases]
    by_method: dict[str, int] = {}
    for tc in test_cases:
        by_method[tc.synthesis_method] = by_method.get(tc.synthesis_method, 0) + 1
    print(f"  {len(test_cases)} held-out cases: {by_method}")

    print("Building leakage-filtered training corpus ...")
    full_corpus = build_retrieval_corpus()
    train_corpus, heldout = split_heldout(full_corpus, test_cases)
    print(f"  {len(train_corpus)} training materials "
          f"({len(full_corpus) - len(train_corpus)} held out)")

    # ------------------------------------------------------------------ #
    # Build requested predictors (lazily; SearchAPI shared where needed)
    # ------------------------------------------------------------------ #
    search_api = None

    def get_search_api():
        nonlocal search_api
        if search_api is None:
            from src.embedding import InputType
            from src.search_api import SearchAPI
            search_api = SearchAPI(input_type=InputType.COMPOSITION,
                                   max_neighbors=100)
        return search_api

    predictors = {}
    for method in args.methods:
        if method == "rule":
            predictors[method] = RulePredictor(train_corpus)
        elif method == "retrieval-jaccard":
            from src.evaluation.baselines import ElementJaccardRetriever
            predictors[method] = RetrievalPredictor(
                ElementJaccardRetriever(corpus=train_corpus), name=method
            )
        elif method == "retrieval-sky":
            from src.evaluation.sky_retriever import SKYRetriever
            predictors[method] = RetrievalPredictor(
                SKYRetriever(get_search_api(), corpus=train_corpus),
                name=method,
            )
        elif method == "llm":
            from src.evaluation.predictors import DirectLLMPredictor
            predictors[method] = CachedPredictor(
                DirectLLMPredictor(model=args.llm_model),
                cache_dir=args.cache_dir,
                retry_failures=args.retry_failures,
            )
        elif method == "sky":
            from src.evaluation.sky_agent_predictor import SKYAgentPredictor
            predictors[method] = CachedPredictor(
                SKYAgentPredictor(
                    model=args.sky_model,
                    exclude_formulas=frozenset(heldout),
                    search_api=get_search_api(),
                ),
                cache_dir=args.cache_dir,
                retry_failures=args.retry_failures,
            )

    # ------------------------------------------------------------------ #
    # Run benchmark
    # ------------------------------------------------------------------ #
    benchmark = PlanningBenchmark(test_cases, verbose=True)
    all_results = {}
    for method, predictor in predictors.items():
        print(f"\n{'=' * 60}\nRunning {method} ...")
        all_results[method] = benchmark.evaluate(predictor, name=method)

    # ------------------------------------------------------------------ #
    # Optional rubric scoring of raw LLM outputs
    # ------------------------------------------------------------------ #
    rubric_scores = {}
    if args.rubric:
        from src.evaluation.llm_eval import RubricEvaluator
        evaluator = RubricEvaluator()
        for method, res in all_results.items():
            texts = [p.raw_text for p in res.predictions if p.raw_text]
            if not texts:
                continue
            print(f"\nRubric-scoring {len(texts)} outputs for {method} ...")
            scores = evaluator.batch_evaluate(texts)
            rubric_scores[method] = {
                "mean_total": sum(s.total for s in scores) / len(scores),
                "n_scored": len(scores),
            }

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #
    for metric in ("element_jaccard", "formula_f1"):
        print(f"\n{'=' * 60}")
        print(f"TABLE 2  —  {metric}  (n={len(test_cases)} held-out cases)")
        print(format_planning_table(all_results, metric=metric))

    print(f"\n{'=' * 60}\nFULL SUMMARY")
    for res in all_results.values():
        print(res.summary_table())
        print()

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        name: {"model": getattr(predictors[name], "inner", predictors[name]).__dict__.get("model"),
               **res.aggregate(),
               "per_case": res.per_case}
        for name, res in all_results.items()
    }

    # Pairwise Wilcoxon per metric, Holm-corrected over each metric's family
    pairwise = {}
    for metric in ("element_jaccard", "formula_f1", "method_accuracy"):
        scores = {n: r.per_case[metric] for n, r in all_results.items()}
        raw = {f"{a}|{b}": v for (a, b), v in wilcoxon_table(scores).items()}
        holm = holm_correction({k: v["p_value"] for k, v in raw.items()})
        pairwise[metric] = {
            k: {**v, "p_holm": holm[k]["p_holm"],
                "significant_holm": holm[k]["significant"]}
            for k, v in raw.items()
        }
    serializable["_pairwise"] = pairwise

    # Temperature MAE on the COMMON subset: cases where every evaluated
    # method produced a scoreable temperature error. Per-method MAEs on
    # different subsets are not comparable.
    per_case_temp = {n: r.per_case["temp_abs_err"] for n, r in all_results.items()}
    n_total = len(test_cases)
    common_idx = [
        i for i in range(n_total)
        if all(errs[i] is not None for errs in per_case_temp.values())
    ]
    serializable["_temperature_common_subset"] = {
        "n_common": len(common_idx),
        "mae": {
            name: (float(sum(errs[i] for i in common_idx) / len(common_idx))
                   if common_idx else None)
            for name, errs in per_case_temp.items()
        },
        "coverage": {
            name: sum(e is not None for e in errs) / n_total
            for name, errs in per_case_temp.items()
        },
    }
    if rubric_scores:
        serializable["_rubric"] = rubric_scores
    serializable["_config"] = {
        "test_set_file": "results/test_set_planning_seed42.json",
        "n_cases": len(test_cases),
        "sky_model": args.sky_model,
        "llm_model": args.llm_model,
        "date": datetime.now(timezone.utc).isoformat(),
    }

    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
