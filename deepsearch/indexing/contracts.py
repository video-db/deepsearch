from __future__ import annotations

from datetime import UTC, datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IndexArtifact(BaseModel):
    index_name: str
    index_id: str
    scene_count: int


class StageTiming(BaseModel):
    stage: str
    duration_ms: int


class ReplacedIndexRef(BaseModel):
    index_name: str
    old_index_id: str
    new_index_id: str


class IndexStats(BaseModel):
    total_scenes: int = 0
    stage_timings: List[StageTiming] = Field(default_factory=list)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    replaced_indexes: List[ReplacedIndexRef] = Field(default_factory=list)


class IndexEvent(BaseModel):
    stage: Literal[
        "extract",
        "transcript",
        "detect",
        "enrich",
        "summarize",
        "write_indexes",
        "manifest",
    ]
    status: Literal["started", "completed", "failed"]
    message: Optional[str] = None
    progress: Optional[float] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IndexStageStatus(BaseModel):
    stage: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = None
    message: Optional[str] = None


class IndexManifest(BaseModel):
    manifest_id: str
    collection_id: str
    video_id: str
    indexes: Dict[str, IndexArtifact]
    stats: IndexStats
    stage_statuses: List[IndexStageStatus]
    warnings: List[str] = Field(default_factory=list)
