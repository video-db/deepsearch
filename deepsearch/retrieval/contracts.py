from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ClipExplain(BaseModel):
    primary_subquery: str = ""
    primary_index: str = ""
    support_subqueries: List[str] = Field(default_factory=list)


class ClipResult(BaseModel):
    video_id: str
    start: float
    end: float
    stream_url: Optional[str] = None
    rank: int = 0
    score: float = 0.0
    validator_status: Optional[str] = None
    explain: ClipExplain = Field(default_factory=ClipExplain)


class PageInfo(BaseModel):
    page_size: int
    cursor: int
    next_cursor: Optional[int] = None
    has_more: bool = False


class ClarificationOption(BaseModel):
    id: str
    label: str


class ClarificationQuestion(BaseModel):
    question_id: str = ""
    text: str = ""
    mode: Literal["text", "mcq"] = "text"
    options: List[ClarificationOption] = Field(default_factory=list)


class UiEvent(BaseModel):
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    session_id: str
    clips: List[ClipResult] = Field(default_factory=list)
    waiting_for: Literal["user_input", "clarification", "none"] = "none"
    clarification: Optional[ClarificationQuestion] = None
    page: PageInfo = Field(default_factory=lambda: PageInfo(page_size=10, cursor=0))
    debug: Optional[Dict[str, Any]] = None
