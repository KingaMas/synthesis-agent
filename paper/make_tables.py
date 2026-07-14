#!/usr/bin/env python
"""
Generate LaTeX tables AND all in-prose numbers for the paper.

Reads   results/benchmark_results.json        (element-SRO retrieval)
        results/transferability_results.json  (formula-SRO + oracle)
        results/planning_results.json         (planning benchmark)
Writes  paper/tables/table1_retrieval.tex
        paper/tables/table2_planning.tex
        paper/numbers.tex                     (named macros)

WORKING RULE (audit 2026-07-14): every number in the manuscript prose must
come from a \\newcommand in numbers.tex — never hand-copied. If a statistic
you need is missing, add a macro here; do not type digits into main.tex.

Run from repo root:  python paper/make_tables.py
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_TABLES = REPO / "paper" / "tables"
NUMBERS = REPO / "paper" / "numbers.tex"

METHOD_LABELS_T2 = {
    "rule": "Rule baseline",
    "retrieval-jaccard": "Retrieval (element Jaccard)",
    "retrieval-sky": "Retrieval (Magpie)",
    "llm": "Direct LLM",
    "sky": "SKY agent",
}

_macros: list[tuple[str, str]] = []


def macro(name: str, value: str):
    if not re.fullmatch(r"[a-zA-Z]+", name):
        raise ValueError(f"macro names must be alphabetic: {name}")
    _macros.append((name, value))


def _ci(entry: dict, digits: int = 3) -> str:
    return (
        f"{entry['mean']:.{digits}f} "
        f"[{entry['ci_lo']:.{digits}f}, {entry['ci_hi']:.{digits}f}]"
    )


# ---------------------------------------------------------------------------
# Table 1: retrieval — formula-SRO (primary), element-SRO (secondary), oracle
#
# CANONICAL SOURCES (do not mix): formula-SRO, oracle, regret, and per-route
# numbers come from transferability_results.json (retrievers called with
# k=5 directly). Element-SRO/MCR/latency come from benchmark_results.json.
# benchmark_results.json also carries a formula_sro column (retrieve at
# k=20, truncate) — rank-fusion retrievers score slightly differently under
# that protocol, so it is NOT used for Table 1.
# ---------------------------------------------------------------------------

def table1() -> str:
    element = json.loads((REPO / "results" / "benchmark_results.json").read_text())
    transfer = json.loads(
        (REPO / "results" / "transferability_results.json").read_text()
    )
    k = transfer["_config"]["k"]

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Retriever & formula-SRO@{k} & element-SRO@{k} & MCR@{k} & ms/query \\",
        r"\midrule",
    ]
    for name, res in element.items():
        if name.startswith("_") or name not in transfer:
            continue
        row = res["per_k"][str(k)]
        timing = res.get("timing", {})
        ms = f"{timing['mean_s'] * 1000:.0f}" if timing else "--"
        lines.append(
            rf"{name} & {_ci(transfer[name])} & {row['sro']['mean']:.3f} & "
            rf"{row['mcr']['mean']:.3f} & {ms} \\"
        )
    lines.append(r"\midrule")
    lines.append(
        rf"\emph{{Oracle ceiling}} & {_ci(transfer['oracle'])} & -- & -- & -- \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]

    # Prose macros for Table-1 discussion
    macro("nRetrievalQueries", str(transfer["_config"]["n_queries"]))
    macro("corpusSize", f"{transfer['_config']['corpus_size']:,}")
    macro("jaccFormulaSro", f"{transfer['Element Jaccard']['mean']:.3f}")
    macro("magpieFormulaSro", f"{transfer['MAGPIE (SKY)']['mean']:.3f}")
    macro("stoichFormulaSro", f"{transfer['Stoich Vector']['mean']:.3f}")
    macro("randomFormulaSro", f"{transfer['Random']['mean']:.3f}")
    macro("hybridFormulaSro", f"{transfer['Hybrid (MAGPIE+Stoich RRF)']['mean']:.3f}")
    macro("oracleFormulaSro", f"{transfer['oracle']['mean']:.3f}")
    macro("jaccMeanRegret", f"{transfer['Element Jaccard']['mean_regret']:.3f}")
    macro(
        "jaccFracLargeRegret",
        f"{transfer['Element Jaccard']['frac_regret_gt_0.2'] * 100:.0f}\\%",
    )
    for route, key in (("solid-state", "SolidState"), ("sol-gel", "SolGel")):
        jr = transfer["Element Jaccard"]["per_route"].get(route)
        orr = transfer["oracle"]["per_route"].get(route)
        if jr and orr:
            macro(f"jacc{key}FormulaSro", f"{jr['mean']:.3f}")
            macro(f"oracle{key}FormulaSro", f"{orr['mean']:.3f}")
            macro(f"gap{key}", f"{orr['mean'] - jr['mean']:.3f}")
    macro("jaccElementSro", f"{element['Element Jaccard']['per_k']['5']['sro']['mean']:.3f}")
    macro("magpieElementSro", f"{element['MAGPIE (SKY)']['per_k']['5']['sro']['mean']:.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: planning — common-subset temperature, per-method coverage
# ---------------------------------------------------------------------------

def table2() -> str:
    data = json.loads((REPO / "results" / "planning_results.json").read_text())
    temp = data["_temperature_common_subset"]

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Precursor Jaccard & Precursor F1 & Method acc. & "
        rf"Temp.\ MAE$^{{a}}$ (\si{{\celsius}}) & Cov. \\",
        r"\midrule",
    ]
    for key, label in METHOD_LABELS_T2.items():
        res = data.get(key)
        if res is None:
            continue
        m = res["metrics"]
        mae = temp["mae"].get(key)
        mae_s = f"{mae:.0f}" if mae is not None else "--"
        lines.append(
            rf"{label} & {_ci(m['element_jaccard'])} & {_ci(m['formula_f1'])} & "
            rf"{m['method_accuracy']['mean']:.2f} & {mae_s} & "
            rf"{temp['coverage'][key]:.2f} \\"
        )
    lines += [
        r"\bottomrule",
        rf"\multicolumn{{6}}{{l}}{{\footnotesize $^a$MAE on the common subset "
        rf"($n={temp['n_common']}$) where all methods produced a scoreable "
        rf"temperature.}}",
        r"\end{tabular}",
    ]

    # Prose macros: point estimates, CIs, Holm-corrected pairwise stats
    n_cases = data["_config"]["n_cases"]
    macro("nPlanningCases", str(n_cases))
    for key, prefix in (
        ("sky", "sky"), ("llm", "llm"), ("rule", "rule"),
        ("retrieval-jaccard", "retrJacc"), ("retrieval-sky", "retrMagpie"),
    ):
        res = data.get(key)
        if res is None:
            continue
        m = res["metrics"]
        macro(f"{prefix}FOne", f"{m['formula_f1']['mean']:.3f}")
        macro(
            f"{prefix}FOneCI",
            f"[{m['formula_f1']['ci_lo']:.3f}, {m['formula_f1']['ci_hi']:.3f}]",
        )
        macro(f"{prefix}ElemJacc", f"{m['element_jaccard']['mean']:.3f}")
        macro(f"{prefix}MethodAcc", f"{m['method_accuracy']['mean']:.2f}")
        macro(f"{prefix}Failures", str(res["n_failures"]))

    pw = data["_pairwise"]["formula_f1"]
    pair_macros = {
        "retrieval-jaccard|sky": "skyVsRetrJacc",
        "rule|sky": "skyVsRule",
        "llm|sky": "skyVsLlm",
    }
    for pair, prefix in pair_macros.items():
        v = pw.get(pair)
        if v is None:
            continue
        macro(f"{prefix}FOnePRaw", f"{v['p_value']:.3f}")
        macro(f"{prefix}FOnePHolm", f"{v['p_holm']:.3f}")
        macro(f"{prefix}FOneEffect", f"{v['effect_size']:.2f}")

    macro("tempCommonN", str(temp["n_common"]))
    for key, prefix in (("rule", "rule"), ("sky", "sky"), ("llm", "llm")):
        mae = temp["mae"].get(key)
        if mae is not None:
            macro(f"{prefix}TempMaeCommon", f"{mae:.0f}")

    return "\n".join(lines)


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "table1_retrieval.tex").write_text(table1() + "\n")
    (OUT_TABLES / "table2_planning.tex").write_text(table2() + "\n")

    header = (
        "% AUTO-GENERATED by paper/make_tables.py — do not edit by hand.\n"
        "% Every number quoted in prose must come from a macro in this file.\n"
    )
    body = "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in _macros
    )
    NUMBERS.write_text(header + body + "\n")
    print(f"Wrote tables to {OUT_TABLES} and {len(_macros)} macros to {NUMBERS}")


if __name__ == "__main__":
    main()
