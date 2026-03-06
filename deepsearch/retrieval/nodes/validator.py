from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Tuple

from langgraph.types import Command

from deepsearch.retrieval.helpers.schema import Plan, JoinedShot
from deepsearch.retrieval.helpers.reason_codes import (
    rc,
    VALIDATOR_SUMMARY,
    VALIDATOR_FEEDBACK,
)
from deepsearch.retrieval.state import GraphState

logger = logging.getLogger(__name__)


class ValidatorLLM:
    def __init__(self, llm, prompts, max_items: int = 40, batch_size: int = 8):
        self.llm = llm
        self.prompts = prompts
        self.max_items = int(max_items)
        self.batch_size = int(batch_size)

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
        self, main_query: str, plan: Plan, candidates: List[JoinedShot], history
    ) -> Tuple[Dict[str, Any], int, int, int]:
        items = [self._to_item(s) for s in candidates[: self.max_items]]
        plan_snapshot = plan.model_dump()
        payload = {
            "plan_snapshot": plan_snapshot,
            "candidates": items,
            "history": history,
        }
        prompt = (
            self.prompts.build_validator_prompt(main_query)
            + "\n\nINPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        data, inp, out, total = self.llm.generate(prompt)
        data = data or {}

        per = data.get("per_shot") or []
        counts = {"pass": 0, "ambiguous": 0, "fail": 0}
        for r in per:
            v = r.get("verdict", "pass")
            if v in counts:
                counts[v] += 1

        result: Dict[str, Any] = {
            "per_shot": per,
            "reason_code": rc(VALIDATOR_SUMMARY, **counts),
        }
        fb = data.get("feedback")
        if counts["pass"] + counts["ambiguous"] == 0 and fb:
            result["feedback"] = {
                "summary": fb.get("summary", "No exact matches."),
                "issues": fb.get("issues")
                if isinstance(fb.get("issues"), list)
                else [],
                "suggested_ops": (
                    fb.get("suggested_ops")
                    if isinstance(fb.get("suggested_ops"), list)
                    else []
                )[:6],
            }
            result["reason_code"] = rc(VALIDATOR_FEEDBACK, **counts)

        return result, inp, out, total


def validator_node(state: GraphState):
    logger.info(f"validator_node: validating {len(state.joined_shots)} shots")
    if state.cfg.debug_mode:
        logger.debug(
            "Validator input main_query=%s candidate_count=%s",
            state.main_query,
            len(state.joined_shots),
        )
    res, _, _, _ = ValidatorLLM(
        state.llm_for("validator"),
        state.prompts,
        max_items=state.cfg.validator_max,
        batch_size=state.cfg.validator_batch_size,
    ).run(
        main_query=state.main_query,
        plan=state.plan,
        candidates=state.joined_shots,
        history=state.history.model_dump(),
    )

    passes = [
        s
        for s, r in zip(state.joined_shots, res.get("per_shot", []))
        if r.get("verdict") == "pass"
    ]
    ambigs = [
        s
        for s, r in zip(state.joined_shots, res.get("per_shot", []))
        if r.get("verdict") == "ambiguous"
    ]
    combined = passes + ambigs
    verdicts = {
        f"{s.video_id}:{s.start}:{s.end}": r.get("verdict", "pass")
        for s, r in zip(state.joined_shots, res.get("per_shot", []))
        if r.get("verdict") in {"pass", "ambiguous"}
    }

    history = state.history
    fb = res.get("feedback")
    if fb:
        history.record_feedback(fb)

    update = {
        "accepted_shots": combined,
        "feedback": fb,
        "history": history.model_dump(),
        "validator_verdicts": verdicts,
    }
    if state.cfg.debug_mode:
        logger.debug("Validator raw_output=%s", json.dumps(res, ensure_ascii=False))
        logger.debug(
            "Validator output feedback=%s",
            json.dumps(fb, ensure_ascii=False) if fb else None,
        )
        logger.debug(
            "Validator accepted_ids=%s",
            [f"{s.video_id}:{s.start}-{s.end}" for s in combined],
        )
    goto = "rerank" if combined else "interpreter"
    logger.info(f"validator_node: {len(combined)} accepted, going to {goto}")
    return Command(update=update, goto=goto)
