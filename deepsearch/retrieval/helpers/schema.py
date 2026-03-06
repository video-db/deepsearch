from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from deepsearch.retrieval.helpers import registry


class Subquery(BaseModel):
    subquery_id: str
    index: List[str]
    q: str
    dialogue: Optional[str] = "false"

    @model_validator(mode="after")
    def _check_index(self):
        if not registry.is_valid_index(self.index):
            raise ValueError(
                f"Unknown index '{self.index}'. Allowed: {registry.allowed_index_names()}"
            )
        return self


class JoinPlan(BaseModel):
    op: str = "OR"
    subqueries: Optional[List[str]] = None
    clauses: Optional[List[List[str]]] = None

    @model_validator(mode="after")
    def _validate(self):
        if self.op not in {"AND", "OR"}:
            raise ValueError("JoinPlan.op must be 'AND' or 'OR'.")
        if not self.subqueries and not self.clauses:
            return self
        if self.subqueries and self.clauses:
            raise ValueError(
                "JoinPlan must use either 'subqueries' or 'clauses', not both."
            )
        if self.subqueries is not None and not all(
            isinstance(x, str) for x in self.subqueries
        ):
            raise ValueError("'subqueries' must be a list of strings.")
        if self.clauses is not None:
            if not all(
                isinstance(cl, list) and all(isinstance(x, str) for x in cl)
                for cl in self.clauses
            ):
                raise ValueError("'clauses' must be a list of string lists.")
        return self


class Plan(BaseModel):
    subqueries: List[Subquery]
    join_plan: JoinPlan
    metadata_filters: Dict[str, List[str]] = Field(default_factory=dict)
    preferences: Dict[str, List[str]] = Field(default_factory=dict)
    fallback_order: List[str] = Field(default_factory=registry.fallback_order)

    @model_validator(mode="after")
    def _validate(self):
        subquery_ids = [sq.subquery_id for sq in self.subqueries]
        if len(subquery_ids) != len(set(subquery_ids)):
            raise ValueError("Each subquery_id must be unique.")
        for k, v in self.metadata_filters.items():
            if k == "scene_index_name":
                continue
            if not registry.is_valid_facet(k):
                raise ValueError(
                    f"Unknown facet '{k}'. Allowed: {registry.allowed_facet_names()}"
                )
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f"Facet '{k}' must be a list of strings.")
        if not self.subqueries:
            raise ValueError("Plan must include at least one subquery.")
        refs = self.join_plan.subqueries or [
            item for clause in (self.join_plan.clauses or []) for item in clause
        ]
        if refs:
            missing = [ref for ref in refs if ref not in set(subquery_ids)]
            if missing:
                raise ValueError(f"JoinPlan references unknown subquery ids: {missing}")
        if len(self.fallback_order) != len(set(self.fallback_order)):
            raise ValueError("fallback_order values must be unique.")
        if any(not registry.is_valid_facet(item) for item in self.fallback_order):
            raise ValueError("fallback_order contains unsupported facets.")
        return self


class Shot(BaseModel):
    video_id: str
    video_title: str
    start: float
    end: float
    text: str
    search_score: Optional[float] = None
    provenance: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Union[List[str], str, bool]] = Field(default_factory=dict)


class JoinedShot(BaseModel):
    video_id: str
    start: float
    end: float
    text: List[Tuple[str, str]]
    primary: Dict[str, Union[List[str], Optional[str]]]
    support_subqueries: List[str] = Field(default_factory=list)
    metadata: Dict[str, Union[List[str], str, bool]] = Field(default_factory=dict)


ShotKey = Tuple[str, float, float]


class History(BaseModel):
    initial_plan: Optional[Plan] = None
    past_qa: List[Dict[str, str]] = Field(default_factory=list)
    delta_ops: List[dict] = Field(default_factory=list)
    feedbacks: List[Dict[str, Any]] = Field(default_factory=list)
    ui_events: List[str] = Field(default_factory=list)

    def record_initial_plan(self, plan: Plan):
        if self.initial_plan is None:
            self.initial_plan = plan.model_copy(deep=True)

    def record_qa(self, question: str, answer: str):
        if question or answer:
            self.past_qa.append({"question": question, "answer": answer})

    def record_delta(self, batch):
        for op in batch.ops:
            self.delta_ops.append(op.model_dump())

    def record_feedback(self, feedback: Dict[str, Any]):
        if feedback:
            self.feedbacks.append(feedback)

    def record_ui_event(self, question: Optional[str], event: str):
        if event:
            self.ui_events.append(json.dumps({"question": question, "event": event}))

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __getitem__(self, item):
        return getattr(self, item)
