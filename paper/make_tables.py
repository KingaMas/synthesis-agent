#!/usr/bin/env python
"""
Generate LaTeX tables for the paper from benchmark result JSONs.

Reads   results/benchmark_results.json   (Table 1, retrieval)
        results/planning_results.json    (Table 2, planning)
Writes  paper/tables/table1_retrieval.tex
        paper/tables/table2_planning.tex

Run from repo root:  python paper/make_tables.py
Re-run whenever the benchmarks are re-executed so the paper always
reflects the committed results.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "tables"

METHOD_LABELS_T2 = {
    "rule": "Rule baseline",
    "retrieval-jaccard": "Retrieval (element Jaccard)",
    "retrieval-sky": "Retrieval (Magpie)",
    "llm": "Direct LLM",
    "sky": "SKY agent",
}


def _ci(entry: dict, digits: int = 3) -> str:
    return (
        f"{entry['mean']:.{digits}f} "
        f"[{entry['ci_lo']:.{digits}f}, {entry['ci_hi']:.{digits}f}]"
    )


def table1(k: int = 5) -> str:
    data = json.loads((REPO / "results" / "benchmark_results.json").read_text())
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Retriever & SRO@{k} & MCR@{k} & NDCG@{k} & ms/query \\",
        r"\midrule",
    ]
    for name, res in data.items():
        if name.startswith("_"):
            continue
        row = res["per_k"][str(k)]
        timing = res.get("timing", {})
        ms = f"{timing['mean_s'] * 1000:.0f}" if timing else "--"
        lines.append(
            rf"{name} & {_ci(row['sro'])} & {row['mcr']['mean']:.3f} & "
            rf"{row['ndcg']['mean']:.3f} & {ms} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table2() -> str:
    data = json.loads((REPO / "results" / "planning_results.json").read_text())
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Precursor Jaccard & Precursor F1 & Method acc. & "
        r"Temp.\ MAE (\si{\celsius}) & Fail \\",
        r"\midrule",
    ]
    for key, label in METHOD_LABELS_T2.items():
        res = data.get(key)
        if res is None:
            continue
        m = res["metrics"]
        t = res["temperature"]
        mae = f"{t['mae']:.0f} ({t['n_scored']})" if t["mae"] is not None else "--"
        lines.append(
            rf"{label} & {_ci(m['element_jaccard'])} & {_ci(m['formula_f1'])} & "
            rf"{m['method_accuracy']['mean']:.2f} & {mae} & {res['n_failures']} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table1_retrieval.tex").write_text(table1() + "\n")
    (OUT / "table2_planning.tex").write_text(table2() + "\n")
    print(f"Wrote tables to {OUT}")


if __name__ == "__main__":
    main()
