"""
LLM output evaluation for SKY.

Three-pronged approach:
1. Automated rubric — 5 dimensions scored 0-2 by GPT-4o-mini.
2. Open-model ablation — run same queries through a local Ollama model.
3. Manual expert validation placeholder — documents expected DOI citations.

Usage
-----
    from src.evaluation.llm_eval import RubricEvaluator
    evaluator = RubricEvaluator()
    scores = evaluator.evaluate(synthesis_output)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Rubric dimensions and scoring
# ---------------------------------------------------------------------------

RUBRIC_DIMENSIONS = [
    "Precursor Specificity",
    "Temperature Specificity",
    "Method Specificity",
    "Physical Reasonableness",
    "Source Grounding",
]

RUBRIC_PROMPT_TEMPLATE = """You are evaluating a materials synthesis recommendation produced by an AI system.
Score the following output on each dimension from 0 to 2:

0 = missing or clearly wrong
1 = partially addressed or vague
2 = specific, accurate, and complete

Dimensions:
1. Precursor Specificity — Are specific precursor compounds named with stoichiometry?
2. Temperature Specificity — Are synthesis temperatures stated in °C or K?
3. Method Specificity — Is the synthesis method (solid-state / hydrothermal / etc.) explicitly named?
4. Physical Reasonableness — Are the stated conditions physically plausible for the target material?
5. Source Grounding — Does the output cite literature sources (DOIs, paper titles, or database IDs)?

Output a JSON object with keys matching the dimension names and integer values 0-2.
Also include a "total" key (sum of all dimensions, max 10) and a brief "comments" string.

Synthesis output to evaluate:
---
{synthesis_output}
---

Return ONLY valid JSON, no preamble."""


@dataclass
class RubricScore:
    precursor_specificity: int
    temperature_specificity: int
    method_specificity: int
    physical_reasonableness: int
    source_grounding: int
    total: int
    comments: str

    @classmethod
    def from_dict(cls, d: dict) -> "RubricScore":
        return cls(
            precursor_specificity=int(d.get("Precursor Specificity", 0)),
            temperature_specificity=int(d.get("Temperature Specificity", 0)),
            method_specificity=int(d.get("Method Specificity", 0)),
            physical_reasonableness=int(d.get("Physical Reasonableness", 0)),
            source_grounding=int(d.get("Source Grounding", 0)),
            total=int(d.get("total", 0)),
            comments=str(d.get("comments", "")),
        )


class RubricEvaluator:
    """Evaluate synthesis outputs using GPT-4o-mini as the judge.

    Requires OPENAI_API_KEY in the environment.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def evaluate(self, synthesis_output: str) -> RubricScore:
        """Score a synthesis output string on all 5 rubric dimensions.

        Args:
            synthesis_output: The full text output from SKY.

        Returns:
            RubricScore with per-dimension and total scores.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required for RubricEvaluator") from e

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MDG_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        client = OpenAI(api_key=api_key)
        prompt = RUBRIC_PROMPT_TEMPLATE.format(synthesis_output=synthesis_output[:4000])

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3]

        scores_dict = json.loads(raw)
        return RubricScore.from_dict(scores_dict)

    def batch_evaluate(self, outputs: list[str]) -> list[RubricScore]:
        """Evaluate a list of synthesis outputs."""
        return [self.evaluate(o) for o in outputs]


# ---------------------------------------------------------------------------
# Open-model ablation via Ollama (local, no API key needed)
# ---------------------------------------------------------------------------

class OllamaRubricEvaluator:
    """Same rubric scoring but routed through a local Ollama model.

    Requires Ollama to be running: https://ollama.com/
    Default model: llama3:70b  (or any model pulled via `ollama pull`)
    """

    def __init__(self, model: str = "llama3:70b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def evaluate(self, synthesis_output: str) -> Optional[RubricScore]:
        """Score output using local Ollama inference."""
        try:
            import urllib.request
            import urllib.error
        except ImportError:
            return None

        prompt = RUBRIC_PROMPT_TEMPLATE.format(synthesis_output=synthesis_output[:4000])
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()

        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            raw = result.get("response", "").strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                if raw.endswith("```"):
                    raw = raw[:-3]
            scores_dict = json.loads(raw)
            return RubricScore.from_dict(scores_dict)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Expert validation reference (Case Study 1 — battery cathodes)
# ---------------------------------------------------------------------------

EXPERT_VALIDATION_REFERENCES = {
    "LiCoO2": [
        {
            "doi": "10.1149/1.2086849",
            "description": "Mizushima et al. 1980 — original LiCoO2 synthesis at 850°C/24h/air",
            "expected_conditions": {
                "temperature_C": 850,
                "time_h": 24,
                "atmosphere": "air",
                "method": "solid-state",
            },
        },
    ],
    "LiFePO4": [
        {
            "doi": "10.1149/1.1378565",
            "description": "Padhi et al. 1997 — LiFePO4 olivine cathode synthesis",
            "expected_conditions": {
                "temperature_C": 800,
                "atmosphere": "inert",
                "method": "solid-state",
            },
        },
    ],
    "LiNiO2": [
        {
            "doi": "10.1016/0022-5088(76)90076-0",
            "description": "Dyer et al. LiNiO2 high-temperature synthesis",
            "expected_conditions": {
                "temperature_C": 700,
                "atmosphere": "oxygen",
                "method": "solid-state",
            },
        },
    ],
}


def check_expert_grounding(synthesis_output: str, material: str) -> dict:
    """Check whether SKY output references the expected DOIs/conditions.

    Args:
        synthesis_output: Text output from SKY.
        material:         Formula string (e.g. 'LiCoO2').

    Returns:
        Dict with 'matched_refs', 'temperature_match', 'method_match'.
    """
    refs = EXPERT_VALIDATION_REFERENCES.get(material, [])
    if not refs:
        return {"message": f"No expert references available for {material}"}

    output_lower = synthesis_output.lower()
    matched_dois: list[str] = []
    for ref in refs:
        doi = ref["doi"]
        if doi in synthesis_output:
            matched_dois.append(doi)

    # Check temperature proximity (within 50°C)
    import re
    temps = [int(t) for t in re.findall(r"(\d{3,4})\s*°?[Cc]", synthesis_output)]
    expected_temps = [
        ref["expected_conditions"].get("temperature_C")
        for ref in refs
        if ref["expected_conditions"].get("temperature_C")
    ]
    temp_match = any(
        abs(t - exp) <= 50
        for t in temps
        for exp in expected_temps
    ) if (temps and expected_temps) else False

    # Check method family
    expected_methods = {ref["expected_conditions"].get("method") for ref in refs}
    method_match = any(m and m.replace("-", " ") in output_lower for m in expected_methods)

    return {
        "material": material,
        "matched_dois": matched_dois,
        "doi_grounding": len(matched_dois) > 0,
        "temperature_match": temp_match,
        "method_match": method_match,
    }
