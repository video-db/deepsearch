from __future__ import annotations

from typing import List

from deepsearch.retrieval.helpers import registry
from deepsearch.retrieval.helpers.schema import Plan, Subquery
from deepsearch.retrieval.helpers.delta import DeltaBatch, DeltaOp, normalize_and_trim, apply_batch


def apply_and_record(plan: Plan, batch: DeltaBatch, history, cfg) -> Plan:
    nb = normalize_and_trim(batch, locks={}, max_ops=getattr(cfg, "max_ops_per_batch", 4))
    return apply_batch(plan, nb)


def apply_batch_or_augment_by_example(plan: Plan, batch: DeltaBatch, text: str, *, llm, history, cfg, prompts) -> Plan:
    example_ops = [op for op in batch.ops if op.op == "by_example"]
    new_plan = plan
    for op in example_ops:
        new_plan = _augment_plan_by_example(new_plan, op, text, llm, prompts)

    drop_ops = [DeltaOp(op="drop_filter", facet=facet) for facet in registry.allowed_facet_names()]
    nb = DeltaBatch(ops=drop_ops, reason_code=batch.reason_code)
    return apply_batch(new_plan, normalize_and_trim(nb, locks={}, max_ops=getattr(cfg, "max_ops_per_batch", 4)))


def _augment_plan_by_example(plan: Plan, op: DeltaOp, text: str, llm, prompts) -> Plan:
    if not op.video_id or op.start is None or op.end is None:
        return plan

    new_plan = plan.model_copy(deep=True)
    used = {sq.subquery_id for sq in new_plan.subqueries}
    i = len(used) + 1
    while f"Q{i}" in used:
        i += 1
    qid = f"Q{i}"

    index = [si for sq in new_plan.subqueries for si in sq.index]

    try:
        gen, _, _, _ = llm.generate(prompts.build_create_query_from_description_prompt() + text)
        q = gen.get("query", "") if isinstance(gen, dict) else ""
    except Exception:
        return plan

    new_plan.subqueries.append(Subquery(subquery_id=qid, index=index, q=q))
    jp = new_plan.join_plan
    if jp.clauses:
        if jp.op == "OR":
            jp.clauses.append([qid])
        else:
            jp.clauses[0].append(qid)
    elif jp.subqueries:
        if jp.op == "OR":
            jp.subqueries.append(qid)
        else:
            jp.clauses = [list(jp.subqueries), [qid]]
            jp.subqueries = None
            jp.op = "OR"
    else:
        jp.subqueries = [qid]
        jp.op = "OR"

    return new_plan
