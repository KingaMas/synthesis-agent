#!/usr/bin/env python
"""
Retrieval-Retro NRE baseline on our splits (T3), stages 1 and 3.

    PYTHONPATH=. python scripts/run_nre_baseline.py prep --protocol internal
    .venv-rr/bin/python scripts/rr_nre_compute.py ...   (stage 2)
    PYTHONPATH=. python scripts/run_nre_baseline.py eval --protocol internal

prep exports target/precursor compositions mapped into Retrieval-Retro's
recovered 82-element vocabulary (element_table.json). eval scores the
delta-G matrix as a retriever (ascending, their convention) alongside
element Jaccard, RR-MPC, and their reference-set union (MPC K=3 + NRE
rest), one Wilcoxon+Holm family. Queries or corpus entries containing
elements outside their vocabulary (or our dummy species like 'A') are
reported, not silently dropped.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MAX_ELS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "eval"])
    ap.add_argument("--protocol", choices=["internal", "frozen"],
                    default="internal")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--data-dir", type=Path, default=Path("results/mpc"))
    args = ap.parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rmb", Path(__file__).parent / "run_mpc_baseline.py")
    rmb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rmb)

    from pymatgen.core import Composition

    from src.evaluation.mpc_data import build_vocabs

    nre_arrays = args.data_dir / f"nre_arrays_{args.protocol}.npz"
    scores_path = args.data_dir / f"nre_scores_{args.protocol}.npz"
    mpc_arrays = args.data_dir / f"mpc_arrays_{args.protocol}.npz"
    mpc_emb = args.data_dir / f"mpc_embeddings_{args.protocol}.npz"

    corpus, query_sets, eval_set = rmb.load_protocol(args.protocol)
    _, prec_vocab = build_vocabs(corpus)
    el_table = json.loads(
        Path("external/rr_dataset/element_table.json").read_text())

    if args.stage == "prep":
        def encode(formulas: list[str]) -> dict[str, np.ndarray]:
            el_ids = np.full((len(formulas), MAX_ELS), -1, dtype=np.int64)
            fracs = np.zeros((len(formulas), MAX_ELS), dtype=np.float32)
            n_els = np.zeros(len(formulas), dtype=np.int64)
            for i, f in enumerate(formulas):
                try:
                    comp = Composition(f).fractional_composition
                except Exception:
                    continue
                items = [(el_table.get(str(el), -1), float(a))
                         for el, a in comp.items()]
                if len(items) > MAX_ELS:
                    continue
                # unknown elements stay as id -1; the compute stage
                # skips those graphs and they score +inf
                for j, (eid, amt) in enumerate(items):
                    el_ids[i, j] = eid
                    fracs[i, j] = amt
                n_els[i] = len(items)
            return {"el_ids": el_ids, "fracs": fracs, "n_els": n_els}

        arrays = {}
        named = {"train": [tc.reduced_formula for tc in corpus],
                 "precursors": prec_vocab,
                 **{n: [tc.reduced_formula for tc in cs]
                    for n, cs in query_sets.items()}}
        for name, formulas in named.items():
            for key, arr in encode(formulas).items():
                arrays[f"{name}_{key}"] = arr
        nre_arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(nre_arrays, **arrays)
        print(f"wrote {nre_arrays}")
        print("next: .venv-rr/bin/python scripts/rr_nre_compute.py "
              f"--nre-arrays {nre_arrays} --mpc-arrays {mpc_arrays} "
              "--rr-repo external/Retrieval-Retro "
              "--checkpoint 'external/rr_dataset/TL_pretrain(formation_exp)_"
              "embedder(graphnetwork)_lr(0.0005)_batch_size(256)_hidden(256)_"
              "seed(0)_.pt' "
              "--element-table external/rr_dataset/element_table.npz "
              f"--out {scores_path}")
        return

    from src.evaluation.baselines import ElementJaccardRetriever
    from src.evaluation.mpc_retriever import (
        PrecomputedEmbeddingRetriever,
        ScoreMatrixRetriever,
        UnionRetriever,
    )
    from src.evaluation.statistics import holm_correction, wilcoxon_table
    from src.evaluation.transferability import evaluate_transferability

    all_queries = query_sets[eval_set]
    blobs = np.load(scores_path)
    scored = blobs[f"{eval_set}_query_scored"]
    # Working rule: comparisons require identical support sets. Restrict
    # ALL retrievers to the NRE-scorable queries (the rest contain
    # elements outside their 82-element vocabulary, e.g. dummy species).
    queries = [tc for tc, ok in zip(all_queries, scored) if ok]
    print(f"common support: {len(queries)}/{len(all_queries)} queries "
          "(rest unmappable into RR element vocabulary)")

    # Two NRE readings. The released calculate_gibbs.py ranks the SIGNED
    # difference Ef(target) - sum Ef(precursors), in which the target
    # only shifts every score by a constant: the ranking is identical
    # for all queries (degenerate). Their shipped reference sets are
    # query-dependent and reproduce under NO variant we tested (signed,
    # |diff|, either label set, target-target; 0/90 overlap with their
    # own checkpoint on their own data), so the released code/checkpoint
    # cannot be what built them. We report the good-faith |delta-G|
    # reading as RR-NRE and the literal released code as RR-NRE-signed.
    score_map = {tc.reduced_formula: row for tc, ok, row in
                 zip(all_queries, scored, blobs[f"{eval_set}_scores"]) if ok}
    abs_map = {f: np.abs(row) for f, row in score_map.items()}
    nre = ScoreMatrixRetriever(corpus, abs_map, smallest=True)
    nre_signed = ScoreMatrixRetriever(corpus, score_map, smallest=True)

    emb = np.load(mpc_emb)
    q_emb = {tc.reduced_formula: e
             for tc, e in zip(all_queries, emb[f"{eval_set}_emb"])}
    mpc = PrecomputedEmbeddingRetriever(corpus, emb["train_emb"], q_emb)

    retrievers = {
        "Element Jaccard": ElementJaccardRetriever(corpus=corpus),
        "RR-MPC": mpc,
        "RR-NRE": nre,
        "RR-NRE-signed": nre_signed,
        "RR-MPC+NRE": UnionRetriever([mpc, nre], quotas=[3, 2]),
    }
    results = evaluate_transferability(retrievers, queries, corpus, k=args.k)

    scores = {name: results[name]["per_query"] for name in retrievers}
    raw = {f"{a}|{b}": v for (a, b), v in wilcoxon_table(scores).items()}
    holm = holm_correction({k_: v["p_value"] for k_, v in raw.items()})
    results["_pairwise"] = {
        k_: {**v, "p_holm": holm[k_]["p_holm"],
             "significant_holm": holm[k_]["significant"]}
        for k_, v in raw.items()
    }
    results["_config"].update({
        "protocol": args.protocol,
        "model": "Retrieval-Retro NRE (pretrained formation-energy "
                 "checkpoint; RR-NRE = |delta-G| good-faith reading, "
                 "RR-NRE-signed = literal released code, degenerate) "
                 "+ MPC union (K=3 MPC + 2 NRE)",
        "reproducibility_note": "shipped year_*_K_3.pt reference sets "
                 "reproduce under no tested variant (0/90 overlap using "
                 "their checkpoint on their data); released "
                 "calculate_gibbs.py ranking is query-independent",
        "support": f"common support {len(queries)}/{len(all_queries)} "
                   "queries (NRE-mappable); all retrievers scored on the "
                   "same subset",
        "n_corpus_usable": int(blobs["corpus_usable"].sum()),
    })

    out = Path(f"results/nre_baseline_{args.protocol}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\n{'=' * 60}")
    for name in retrievers:
        r = results[name]
        sg = r["per_route"].get("sol-gel", {})
        print(f"{name:16s} fSRO@{args.k}={r['mean']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"sol-gel={sg.get('mean', float('nan')):.3f}  "
              f"regret={r['mean_regret']:.3f}")
    print(f"{'Oracle':16s} fSRO@{args.k}={results['oracle']['mean']:.3f}")
    for pair, v in results["_pairwise"].items():
        print(f"  {pair}: p_holm={v['p_holm']:.4f} sig={v['significant_holm']}")
    print(f"\nresults saved to {out}")


if __name__ == "__main__":
    main()
