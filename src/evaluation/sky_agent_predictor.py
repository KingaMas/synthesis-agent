"""
SKY agent predictor for the synthesis planning benchmark.

A benchmark variant of the SKY agent: the same architecture (OpenAI Agents
SDK tool loop over similarity search + recipe lookup) but with

- leakage-aware recipe lookup (RecipeStore built with exclude_formulas),
- composition-only similarity search (no structure H5 / MP API needed),
- a JSON final answer instead of the interactive report format, and
- no HTML-report generation step.

The production agent (sky/core/synthesis_agent.py SKYSynthesisAgent) keeps
its own prompt and full toolset; this module only rewires the tools that
would either leak held-out recipes or fail in a batch environment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from pymatgen.core import Composition

from src.evaluation.planning import RecipePrediction
from src.evaluation.predictors import (
    PREDICTION_JSON_SCHEMA,
    RecipeStore,
    parse_prediction_json,
)
from src.evaluation.test_set_builder import TestCase

AGENT_INSTRUCTIONS = """You are SKY, an expert materials synthesis specialist.
Your task: predict a realistic laboratory synthesis recipe for a target
material that has NO published recipe.

Strategy:
1. Use search_similar_materials to find compositionally similar materials.
2. Use get_known_recipes on the most similar materials to gather real
   published recipes (the target itself has none).
3. Adapt the closest recipes to the target: substitute precursors for the
   differing elements, keep method and conditions from the best-matching
   source.

Your FINAL message must be ONLY a JSON object, no prose:
{schema}

precursors: commercially available starting compounds.
method: exactly one of "solid-state" or "sol-gel".
max_heating_temperature_C: highest calcination/sintering temperature in
Celsius, or null if you cannot estimate it."""

QUERY_TEMPLATE = (
    'Predict a synthesis recipe for the target material "{formula}". '
    "Search for similar materials, gather their known recipes, then answer "
    "with the JSON object only."
)


class SKYAgentPredictor:
    """Full SKY-architecture predictor (tool loop) for the benchmark."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        exclude_formulas: frozenset[str] = frozenset(),
        search_api=None,
        recipes_path: Optional[Path] = None,
        max_turns: int = 12,
    ):
        self.model = model
        self.name = f"sky[{model}]"
        self.exclude_formulas = frozenset(exclude_formulas)
        self.recipes_path = recipes_path
        self.max_turns = max_turns
        self._search_api = search_api
        self._agent = None

        # Mirror SKYSynthesisAgent's key handling
        mdg_key = os.getenv("OPENAI_MDG_API_KEY")
        if mdg_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = mdg_key

    # -- lazy construction (RecipeStore + H5 load are expensive) ---------

    def _get_search_api(self):
        if self._search_api is None:
            from src.embedding import InputType
            from src.search_api import SearchAPI

            self._search_api = SearchAPI(
                input_type=InputType.COMPOSITION, max_neighbors=100
            )
        return self._search_api

    def _get_agent(self):
        if self._agent is not None:
            return self._agent

        from agents import Agent, function_tool

        store = RecipeStore(self.recipes_path, self.exclude_formulas)
        search_api = self._get_search_api()

        @function_tool
        def search_similar_materials(formula: str, top_n: int = 10) -> str:
            """Find materials compositionally similar to a formula.

            Args:
                formula: Target composition, e.g. "BaTiO3".
                top_n: Number of similar materials to return.
            """
            try:
                neighbors = search_api.query(
                    Composition(formula), n_neighbors=top_n
                )
                return json.dumps(
                    [
                        {
                            "formula": n.formula,
                            "distance": round(n.distance, 3),
                            "confidence": round(n.confidence, 4),
                        }
                        for n in neighbors
                    ]
                )
            except Exception as e:  # noqa: BLE001 - tool errors go to the model
                return json.dumps({"error": str(e)})

        @function_tool
        def get_known_recipes(formula: str) -> str:
            """Look up published synthesis recipes for a material formula.

            Args:
                formula: Composition to look up, e.g. "SrTiO3".
            """
            from sky.core.synthesis_agent import _summarize_recipe

            recipes = store.lookup(formula)[:3]
            if not recipes:
                return json.dumps(
                    {"formula": formula, "recipes": [], "note": "no known recipe"}
                )
            return json.dumps(
                {"formula": formula, "recipes": [_summarize_recipe(r) for r in recipes]},
                default=str,
            )

        @function_tool
        def analyze_synthesis_parameters(synthesis_text: str) -> str:
            """Extract temperatures, times, methods and atmosphere from recipe text.

            Args:
                synthesis_text: Free-text synthesis description.
            """
            from sky.core.synthesis_agent import analyze_synthesis_parameters_impl

            return analyze_synthesis_parameters_impl(synthesis_text)

        self._agent = Agent(
            name="SKY_PlanningBenchmark",
            instructions=AGENT_INSTRUCTIONS.format(schema=PREDICTION_JSON_SCHEMA),
            model=self.model,
            tools=[
                search_similar_materials,
                get_known_recipes,
                analyze_synthesis_parameters,
            ],
        )
        return self._agent

    # -- prediction ------------------------------------------------------

    def predict(self, case: TestCase) -> RecipePrediction:
        from agents import Runner

        agent = self._get_agent()
        start = time.perf_counter()
        result = Runner.run_sync(
            agent,
            input=QUERY_TEMPLATE.format(formula=case.reduced_formula),
            max_turns=self.max_turns,
        )
        latency = time.perf_counter() - start

        pred = parse_prediction_json(
            result.final_output or "", case.reduced_formula
        )
        pred.latency_s = latency
        pred.model = self.model
        return pred
