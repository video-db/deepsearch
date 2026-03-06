from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

from langgraph.types import Command
from pydantic import ValidationError

from deepsearch.retrieval.helpers.delta import DeltaBatch
from deepsearch.retrieval.helpers.reason_codes import rc, INTENT_ROUTE
from deepsearch.retrieval.helpers.batch_apply import (
    apply_and_record,
    apply_batch_or_augment_by_example,
)
from deepsearch.retrieval.state import GraphState

logger = logging.getLogger(__name__)


class IntentInterpreterLLM:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def run(
        self,
        user_text: Optional[str],
        ui_event: Optional[dict],
        context: Dict[str, Any],
    ):
        payload = {"user_text": user_text, "ui_event": ui_event, "context": context}
        prompt = (
            self.prompts.build_interpreter_prompt()
            + "\n\nINPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        data, inp, out, total = self.llm.generate(prompt)
        data = data or {}
        if "reason_code" not in data:
            data["reason_code"] = rc(INTENT_ROUTE, path=data.get("intent"))
        return data, inp, out, total


def interpreter_node(state: GraphState):
    logger.info("interpreter_node: interpreting user input")
    history = state.history
    ui_event_payload = state.ui_event.model_dump() if state.ui_event else None

    if state.clarify_question and state.ui_event:
        history.record_ui_event(
            state.clarify_question.get("question"),
            json.dumps(ui_event_payload, ensure_ascii=False),
        )
    if state.clarify_question and state.user_text:
        history.record_qa(state.clarify_question.get("question", ""), state.user_text)

    if state.ui_event and state.ui_event.type == "by_example":
        payload = state.ui_event.payload
        batch = DeltaBatch(
            ops=[
                DeltaBatch.model_validate(
                    {"ops": [{"op": "by_example", **payload}]}
                ).ops[0]
            ]
        )
        new_plan = apply_batch_or_augment_by_example(
            state.plan,
            batch,
            payload.get("text", ""),
            llm=state.llm_for("interpreter"),
            history=history,
            cfg=state.cfg,
            prompts=state.prompts,
        )
        history.record_delta(batch)
        return Command(
            update={
                "plan": new_plan,
                "history": history.model_dump(),
                "intent": "by_example",
            },
            goto="search_join",
        )

    context = {
        "main_query": state.main_query,
        "plan_snapshot": state.plan.model_dump(),
        "last_preview": [s.model_dump() for s in state.last_preview],
        "validator_feedback": state.feedback,
        "history": history.model_dump(),
        "past_qa": history.past_qa if history else [],
    }
    if state.cfg.debug_mode:
        logger.debug(
            "Interpreter context=%s ui_event=%s user_text=%s",
            json.dumps(context, ensure_ascii=False),
            json.dumps(ui_event_payload, ensure_ascii=False)
            if ui_event_payload
            else None,
            state.user_text,
        )

    data, _, _, _ = IntentInterpreterLLM(
        state.llm_for("interpreter"), state.prompts
    ).run(state.user_text, ui_event_payload, context)

    plan = state.plan
    cfg = state.cfg
    update: Dict[str, Any] = {}

    if data.get("batch") or data.get("delta"):
        raw = data.get("batch") or data.get("delta")
        try:
            batch = (
                DeltaBatch.model_validate(raw)
                if not isinstance(raw, DeltaBatch)
                else raw
            )
        except ValidationError as exc:
            logger.warning(
                "Interpreter produced invalid delta batch; asking clarification: %s",
                exc,
            )
            update["clarify_question"] = {
                "question_id": "clarify-invalid-delta",
                "text": "I could not parse that refinement request. Please rephrase it.",
                "mode": "text",
                "options": [],
            }
            update["intent"] = "clarify"
            update["history"] = history.model_dump()
            return Command(update=update, goto="clarify_pause")
        if any(op.op == "by_example" for op in batch.ops):
            example_text = next((op.text for op in batch.ops if op.text), "")
            new_plan = apply_batch_or_augment_by_example(
                plan,
                batch,
                example_text,
                llm=state.llm_for("interpreter"),
                history=history,
                cfg=cfg,
                prompts=state.prompts,
            )
        else:
            new_plan = apply_and_record(plan, batch, history, cfg)
        history.record_delta(batch)
        update["plan"] = new_plan
        goto = "search_join"
    elif data.get("clarify_question"):
        cq = data["clarify_question"]
        update["clarify_question"] = {
            "question_id": cq.get("question_id", "clarify-1"),
            "text": cq.get("text") or cq.get("question", ""),
            "mode": "mcq" if cq.get("mode") == "mcq" else "text",
            "options": cq.get("options")
            or [
                {"id": choice.get("id", ""), "label": choice.get("label", "")}
                for choice in cq.get("choices", [])
            ],
        }
        goto = "clarify_pause"
    elif data.get("intent") == "show_more":
        goto = "preview_pause"
    else:
        goto = "preview_pause"

    update["intent"] = data.get("intent")
    update["history"] = history.model_dump()
    if state.cfg.debug_mode:
        logger.debug("Interpreter output=%s", json.dumps(data, ensure_ascii=False))
    logger.info(f"interpreter_node: intent={data.get('intent')}, going to {goto}")
    return Command(update=update, goto=goto)
