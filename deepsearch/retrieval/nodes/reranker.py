from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from langgraph.types import Command

from deepsearch.retrieval.helpers.schema import JoinedShot
from deepsearch.retrieval.state import GraphState

logger = logging.getLogger(__name__)


class RerankerLLM:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def run(self, shots: List[JoinedShot], user_query: str, plan, history, past_qa):
        ids = [f"{s.video_id}:{s.start}-{s.end}" for s in shots]

        if not self.llm:
            ranked = sorted(
                shots,
                key=lambda s: (
                    len(s.support_subqueries),
                    float(s.primary.get("search_score") or 0.0),
                ),
                reverse=True,
            )
            return {"ranking": ranked}, 0, 0, 0

        payload = {
            "shots": [
                {
                    "id": ids[i],
                    "primary_index": shots[i].primary.get("index"),
                    "support_count": len(shots[i].support_subqueries),
                    "metadata": shots[i].metadata,
                    "score": float(shots[i].primary.get("search_score") or 0.0),
                }
                for i in range(len(shots))
            ],
            "prefs": plan.preferences or {},
            "main_query": user_query,
            "user_query_latest": [(subq.q, subq.dialogue) for subq in plan.subqueries],
            "history": history,
            "past_questions_and_answers": past_qa,
        }
        prompt = (
            self.prompts.build_reranker_prompt()
            + "\n\nINPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        data, inp, out, total = self.llm.generate(prompt)
        data = data or {}
        order = data.get("ranking") or []

        if len(order) != len(ids) or set(order) != set(ids):
            ordered = sorted(
                range(len(shots)),
                key=lambda i: (
                    len(shots[i].support_subqueries),
                    float(shots[i].primary.get("search_score") or 0.0),
                ),
                reverse=True,
            )
        else:
            idx_map = {id_: i for i, id_ in enumerate(ids)}
            ordered = [idx_map[id_] for id_ in order]

        return {"ranking": [shots[i] for i in ordered]}, inp, out, total


def rerank_node(state: GraphState):
    logger.info(f"rerank_node: reranking {len(state.accepted_shots)} shots")
    if state.cfg.debug_mode:
        logger.debug(
            "Reranker input accepted_count=%s main_query=%s",
            len(state.accepted_shots),
            state.main_query,
        )
    res, _, _, _ = RerankerLLM(state.llm_for("reranker"), state.prompts).run(
        shots=state.accepted_shots,
        user_query=state.main_query,
        plan=state.plan,
        history=state.history.model_dump(),
        past_qa=state.history.past_qa if state.history else [],
    )
    if state.cfg.debug_mode:
        ranked = res.get("ranking", [])
        logger.debug("Reranker output ranked_count=%s", len(ranked))
        logger.debug(
            "Reranker ordered_ids=%s",
            [f"{s.video_id}:{s.start}-{s.end}" for s in ranked],
        )
    return Command(update={"ranked_shots": res["ranking"]}, goto="preview_pause")
