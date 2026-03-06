from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from langgraph.types import Command

from deepsearch.retrieval.helpers.schema import Plan, Subquery, JoinPlan, History
from deepsearch.retrieval.helpers import registry
from deepsearch.retrieval.state import GraphState

logger = logging.getLogger(__name__)


class PlanInitLLM:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def run(self, user_query: str) -> Tuple[Plan, Dict[str, int]]:
        obj_inp = obj_out = obj_total = 0

        prompt = self.prompts.build_plan_init_prompt(user_query)
        if getattr(self.prompts, "debug_mode", False):
            logger.debug("PlanInit input query=%s", user_query)
        data, plan_inp, plan_out, plan_total = self.llm.generate(prompt)
        data = data or {}

        subqs = [Subquery.model_validate(sq) for sq in data.get("subqueries", [])]
        jp_raw = data.get("join_plan") or {
            "op": "OR",
            "subqueries": [sq.subquery_id for sq in subqs],
        }
        join_plan = JoinPlan.model_validate(jp_raw)
        mfilters: Dict[str, List[str]] = data.get("metadata_filters") or {}
        prefs: Dict[str, List[str]] = data.get("preferences") or {}
        fb_order = self._normalize_fallback_order(
            data.get("fallback_order") or registry.fallback_order()
        )

        for sq in subqs:
            if "transcript" in sq.index or "topic" in sq.index:
                sq.dialogue = "true"

        if mfilters.get("objects"):
            obj_names, obj_inp, obj_out, obj_total = self._resolve_objects(
                mfilters["objects"]
            )
            mfilters["objects"] = [
                str(x)
                for item in (obj_names or [])
                for x in (item if isinstance(item, (list, tuple)) else [item])
                if x is not None
            ]

        # Strip facets not in registry
        mfilters = {k: v for k, v in mfilters.items() if registry.is_valid_facet(k)}

        for key in list(mfilters.keys()):
            try:
                mfilters[key] = sorted(set(mfilters.get(key) or []))
            except Exception:
                pass

        plan = Plan(
            subqueries=subqs,
            join_plan=join_plan,
            metadata_filters=mfilters,
            preferences=prefs,
            fallback_order=fb_order,
        )

        tokens = {
            "plan_input_tokens": plan_inp,
            "plan_output_tokens": plan_out,
            "plan_total_tokens": plan_total,
            "object_names_input_tokens": obj_inp,
            "object_names_output_tokens": obj_out,
            "object_names_total_tokens": obj_total,
        }
        return plan, tokens

    def _resolve_objects(
        self, query_objects: List[str]
    ) -> Tuple[List[str], int, int, int]:
        data, inp, out, total = self.llm.generate(
            self.prompts.build_object_names_prompt(query_objects)
        )
        data = data or {}
        return data.get("object_names") or [], inp, out, total

    def _normalize_fallback_order(self, fallback_order: List[str]) -> List[str]:
        normalized: List[str] = []
        for item in fallback_order or []:
            candidate = str(item).strip().lower()
            if registry.is_valid_facet(candidate) and candidate not in normalized:
                normalized.append(candidate)
        return normalized or registry.fallback_order()


def plan_init_node(state: GraphState):
    logger.info("plan_init_node: generating plan")
    plan, tokens = PlanInitLLM(state.llm_for("planner"), state.prompts).run(
        state.main_query
    )
    history = state.history if state.history else History()
    history.record_initial_plan(plan)

    update = {"plan": plan, "history": history}
    if state.cfg.debug_mode:
        logger.debug("PlanInit output plan=%s", plan.model_dump())
    logger.info(
        f"plan_init_node: plan generated with {len(plan.subqueries)} subqueries"
    )
    return Command(update=update, goto="search_join")
