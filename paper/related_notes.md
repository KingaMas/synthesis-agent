# Closest prior work per contribution claim (audit rule 5)

Every contribution bullet in the paper needs an entry here: closest prior
work + explicit delta. Sourced from the 2026-07-14 literature review.

## Claim: formula-level transferability benchmark with oracle calibration

- **Closest:** Retrieval-Retro (NeurIPS 2024) — retrieval over 33k recipes
  with MPC + thermodynamic NRE retrievers, year-based splits, implicit
  precursor extraction. He et al., Sci. Adv. 9, eadg8180 (2023) —
  PrecursorSelector, learned materials similarity, solid-state only,
  top-5 success ≥ 82%.
- **Delta:** neither measures whether retrieved neighbours' recipes are
  transferable on a non-circular metric, reports the oracle ceiling /
  regret, calibrates against trivial baselines (raw element Jaccard), or
  covers a multi-route corpus. We do all four; we must run BOTH systems
  inside our harness (T3) before claiming superiority of anything.

## Claim: route-stratified failure analysis (sol-gel gap, precursor-form selection)

- **Closest:** none found that stratifies retrieval failure by synthesis
  route; solid-state-only corpora (He et al., Retrieval-Retro,
  ElemwiseRetro, Retro-Rank-In) cannot express the question.
- **Delta:** we show the oracle gap concentrates in sol-gel and that
  route-conditioning alone closes only ~14% of it (route-oracle ablation,
  audit session; TODO reimplement in src/evaluation/ before citing) —
  the remainder is precursor-form selection (nitrate/alkoxide/citrate vs
  oxide/carbonate, hydrates, chelators).

## Claim: agent/planning evaluation with leakage guard and corrected stats

- **Closest:** ICLR 2026 augmentation benchmark (674 targets): RAG is the
  only consistently helpful augmentation (77.0 → 83.5% top-10 precursor
  accuracy); multi-agent workflows often hurt. MSP-LLM (arXiv 2602.07543)
  predicts precursors + operation sequences — weakens any "full recipe"
  novelty claim. Prein et al. 2025: off-the-shelf LMs, top-1 precursor
  accuracy 53.8%, calcination/sintering temperature MAE < 126 °C.
- **Delta:** our contribution is NOT the agent (consistent with ICLR 2026,
  our agent's F1 edge is not significant after Holm at n=100); it is the
  evaluation protocol: held-out leakage guard, trivial-baseline floors,
  identical-support-set comparisons. Temperature numbers must not be
  compared to Prein et al. without protocol reconciliation (they score
  per-operation temps on solid-state; we score max-heating across mixed
  routes with regex extraction) — see T5 before any temperature claim.

## Claim: LLM rows (contamination caveat)

- gpt-4o-mini has plausibly seen the Kononova corpus in training. Until a
  contamination probe (verbatim recipe-completion on held-out targets) is
  run, LLM rows carry a contamination note and are excluded from core
  claims.

## Agent-demo literature (context, not competition)

- Coscientist (Nature 2023), ChemCrow (Nat. Mach. Intell. 2024), A-Lab
  (Nature 2023), ChemAgents, Chemist-X, AutoLabs, LLM-RDF: demonstration-
  evaluated, not benchmarked against baselines. SKY's delta is evaluation
  rigor, not agent capability.
