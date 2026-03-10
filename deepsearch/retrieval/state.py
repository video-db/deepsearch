from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from deepsearch.config.schema import RetrievalConfig
from deepsearch.retrieval.helpers.schema import JoinedShot, Plan, History
from deepsearch.retrieval.helpers.delta import DeltaBatch
from deepsearch.retrieval.contracts import ClipResult, UiEvent


class GraphState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: Optional[str] = None
    collection_id: str = ""
    collection: Any = None
    video_id: Optional[str] = None
    cfg: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llms: Dict[str, Any] = Field(default_factory=dict)
    prompts: Any = None
    search_client: Any = None

    main_query: str = ""
    history: Optional[History] = None
    plan: Optional[Plan] = None

    clarify_question: Optional[Dict[str, Any]] = None

    joined_shots: List[JoinedShot] = Field(default_factory=list)
    accepted_shots: Optional[List[JoinedShot]] = None
    feedback: Optional[Dict[str, Any]] = None
    ranked_shots: List[JoinedShot] = Field(default_factory=list)
    validator_verdicts: Dict[str, str] = Field(default_factory=dict)

    user_text: Optional[str] = None
    ui_event: Optional[UiEvent] = None

    intent: Optional[str] = None
    batch: Optional[DeltaBatch] = None
    by_index: Optional[str] = None

    paused_for: Optional[str] = None
    ranked_cache: List[ClipResult] = Field(default_factory=list)
    page_cursor: int = 0
    page_size: int = 10
    last_preview: List[ClipResult] = Field(default_factory=list)

    def llm_for(self, task: str):
        return self.llms[task]
