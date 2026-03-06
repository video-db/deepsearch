from __future__ import annotations

from typing import Any, Dict, List


def rc(code: str, **args: Any) -> Dict[str, Any]:
    return {"code": code, "args": args}


PLAN_INIT = "PLAN_INIT"
SEARCH_RUN = "SEARCH_RUN"
INDEX_SET = "INDEX_SET"
JOIN_RESULT = "JOIN_RESULT"
JOIN_ZERO = "JOIN_ZERO"

VALIDATOR_SUMMARY = "VALIDATOR_SUMMARY"
VALIDATOR_FEEDBACK = "VALIDATOR_FEEDBACK"

RERANK_PREF = "RERANK_PREF"
BATCH_TRIMMED = "BATCH_TRIMMED"
BATCH_FROM_USER = "BATCH_FROM_USER"
AUTO_FIX_BUILT = "AUTO_FIX_BUILT"
INTENT_ROUTE = "INTENT_ROUTE"
DELTA_FROM_NONE = "DELTA_FROM_NONE"

_BLOCK_CODES: Dict[str, List[str]] = {
    "validator": [VALIDATOR_SUMMARY, VALIDATOR_FEEDBACK],
    "reranker": [RERANK_PREF],
    "interpreter": [BATCH_FROM_USER, INTENT_ROUTE],
    "none": [DELTA_FROM_NONE],
    "join": [JOIN_RESULT, JOIN_ZERO],
    "plan": [PLAN_INIT],
}


def codes_for(block: str) -> List[str]:
    return _BLOCK_CODES.get(block, [])
