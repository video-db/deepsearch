from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from deepsearch.retrieval.helpers import registry
from deepsearch.retrieval.helpers.schema import Plan


class DeltaOp(BaseModel):
    op: Literal[
        "set_filter",
        "merge_filter",
        "drop_filter",
        "drop_values",
        "set_subquery",
        "merge_indexes",
        "set_join_plan",
        "by_example",
    ]
    facet: Optional[str] = None
    values: Optional[List[str]] = None
    names: Optional[List[str]] = None
    name: Optional[str] = None
    indexes_to_add: Optional[List[str]] = None
    subquery_id: Optional[str] = None
    q: Optional[str] = None
    video_id: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    join_plan: Optional[Dict] = None
    value: Optional[str] = None
    text: Optional[str] = None


class DeltaBatch(BaseModel):
    ops: List[DeltaOp] = Field(default_factory=list)
    reason_code: Dict[str, Any] = Field(default_factory=dict)


def _key(op: DeltaOp) -> Tuple:
    return (
        op.op,
        op.facet,
        tuple(op.values or []),
        tuple(op.names or []),
        tuple(op.indexes_to_add or []),
        op.name,
        op.subquery_id,
        op.q,
        op.video_id,
        op.start,
        op.end,
        op.text,
    )


def normalize_and_trim(
    batch: DeltaBatch, locks: Dict[str, bool], max_ops: int
) -> DeltaBatch:
    filtered: List[DeltaOp] = []
    seen = set()

    for op in batch.ops:
        if op.facet and op.op in {
            "drop_filter",
            "set_filter",
            "merge_filter",
            "drop_values",
        }:
            if locks.get(op.facet):
                continue
            if not registry.is_valid_facet(op.facet):
                continue
            if op.op == "drop_values" and not registry.FACETS[op.facet].get(
                "per_value_relax", False
            ):
                continue

        if op.op == "merge_indexes" and op.indexes_to_add is not None:
            op.indexes_to_add = [
                idx
                for idx in dict.fromkeys(op.indexes_to_add)
                if isinstance(idx, str) and idx.strip()
            ]
            if not op.indexes_to_add:
                continue

        k = _key(op)
        if k in seen:
            continue
        seen.add(k)
        filtered.append(op)

    by_facet: Dict[str, List[str]] = {}
    rest: List[DeltaOp] = []
    for op in filtered:
        if op.op == "drop_values" and op.facet and (op.names or op.name):
            names = list(op.names or []) + ([op.name] if op.name else [])
            by_facet.setdefault(op.facet, []).extend(names)
        else:
            rest.append(op)

    for facet, names in by_facet.items():
        rest.append(DeltaOp(op="drop_values", facet=facet, names=sorted(set(names))))

    applied = rest[:max_ops]
    rc = batch.reason_code
    if len(rest) > max_ops:
        rc = {
            "code": "BATCH_TRIMMED",
            "args": {"applied": len(applied), "deferred": len(rest) - len(applied)},
        }
    return DeltaBatch(ops=applied, reason_code=rc)


def apply_batch(plan: Plan, batch: DeltaBatch) -> Plan:
    new_plan = plan.model_copy(deep=True)

    for op in batch.ops:
        if op.op == "set_filter" and op.facet and op.values is not None:
            new_plan.metadata_filters[op.facet] = list(op.values)
        elif op.op == "merge_filter" and op.facet and op.values:
            cur = new_plan.metadata_filters.get(op.facet, [])
            new_plan.metadata_filters[op.facet] = sorted(set(cur + op.values))
        elif op.op == "drop_filter" and op.facet:
            new_plan.metadata_filters[op.facet] = []
        elif op.op == "drop_values" and op.facet and (op.names or op.name):
            names = list(op.names or []) + ([op.name] if op.name else [])
            cur = new_plan.metadata_filters.get(op.facet, [])
            new_plan.metadata_filters[op.facet] = [
                v for v in cur if v not in set(names)
            ]
        elif op.op == "set_subquery" and op.subquery_id and op.q is not None:
            for sq in new_plan.subqueries:
                if sq.subquery_id == op.subquery_id:
                    sq.q = op.q
        elif op.op == "by_example":
            pass
        elif op.op == "set_join_plan" and op.join_plan:
            from deepsearch.retrieval.helpers.schema import JoinPlan

            new_plan.join_plan = JoinPlan.model_validate(op.join_plan)
        elif op.op == "merge_indexes" and op.subquery_id and op.indexes_to_add:
            valid_indexes = [
                idx for idx in op.indexes_to_add if registry.is_valid_index(idx)
            ]
            if not valid_indexes:
                continue
            for sq in new_plan.subqueries:
                if sq.subquery_id == op.subquery_id:
                    cur = list(sq.index)
                    for idx in valid_indexes:
                        if idx not in cur:
                            cur.append(idx)
                    sq.index = cur
                    break

    return new_plan
