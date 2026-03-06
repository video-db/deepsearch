from __future__ import annotations

import json
import logging
from pydantic import ValidationError
from typing import Any, Dict, List, Optional, Tuple

from langgraph.types import Command

from deepsearch.retrieval.helpers.schema import Plan, JoinedShot
from deepsearch.retrieval.helpers.batch_apply import (
    apply_and_record,
    apply_batch_or_augment_by_example,
)
from deepsearch.retrieval.helpers.delta import DeltaBatch
from deepsearch.retrieval.state import GraphState

logger = logging.getLogger(__name__)


class NoneAnalyzerLLM:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def _to_item(self, s: JoinedShot) -> Dict[str, Any]:
        return {
            "video_id": s.video_id,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "primary": {
                "index": s.primary.get("index"),
                "subquery_id": s.primary.get("subquery_id"),
                "variant_id": s.primary.get("variant_id"),
            },
            "support_subqueries": s.support_subqueries,
            "metadata": s.metadata,
        }

    def run(
        self,
        main_query: str,
        plan: Plan,
        displayed: List[JoinedShot],
        history: Dict,
        past_qa: Optional[List] = None,
    ):
        payload = {
            "main_query": main_query,
            "plan_snapshot": plan.model_dump(),
            "displayed": [self._to_item(s) for s in displayed],
            "history": history,
            "past_qa": past_qa or [],
        }
        prompt = (
            self.prompts.build_none_prompt()
            + "\n\nINPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        return self.llm.generate(prompt)


def none_analyzer_node(state: GraphState):
    logger.info("none_analyzer_node: analyzing empty results")
    if state.cfg.debug_mode:
        logger.debug(
            "NoneAnalyzer input main_query=%s displayed_count=%s",
            state.main_query,
            len(state.ranked_shots),
        )
    data, _, _, _ = NoneAnalyzerLLM(state.llm_for("none_analyzer"), state.prompts).run(
        main_query=state.main_query,
        plan=state.plan,
        displayed=state.ranked_shots,
        history=state.history.model_dump(),
        past_qa=state.history.past_qa if state.history else [],
    )
    data = data or {}
    plan = state.plan
    history = state.history

    if data.get("batch"):
        try:
            batch = (
                DeltaBatch.model_validate(data["batch"])
                if not isinstance(data["batch"], DeltaBatch)
                else data["batch"]
            )
        except ValidationError as exc:
            logger.warning(
                "NoneAnalyzer produced invalid delta batch; asking clarification: %s",
                exc,
            )
            update = {
                "clarify_question": {
                    "question_id": "clarify-invalid-none-delta",
                    "text": "I could not parse that refinement. Please rephrase it.",
                    "mode": "text",
                    "options": [],
                }
            }
            return Command(update=update, goto="clarify_pause")
        if any(op.op == "by_example" for op in batch.ops):
            example_text = next((op.text for op in batch.ops if op.text), "")
            new_plan = apply_batch_or_augment_by_example(
                plan,
                batch,
                example_text,
                llm=state.llm_for("interpreter"),
                history=history,
                cfg=state.cfg,
                prompts=state.prompts,
            )
        else:
            new_plan = apply_and_record(plan, batch, history, state.cfg)
        history.record_delta(batch)
        update = {"plan": new_plan, "history": history.model_dump()}
        goto = "search_join"
    elif data.get("clarify_question"):
        update = {"clarify_question": data["clarify_question"]}
        goto = "clarify_pause"
    else:
        raise ValueError("NoneAnalyzerLLM returned neither batch nor clarify_question")

    if state.cfg.debug_mode:
        logger.debug("NoneAnalyzer output=%s", json.dumps(data, ensure_ascii=False))
    logger.info(f"none_analyzer_node: going to {goto}")
    return Command(update=update, goto=goto)
