from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field


class LLMIndexingModelsConfig(BaseModel):
    scene_enrichment: str = "o3"
    subplot_summary: str = "o3"
    final_summary: str = "o3"


class LLMRetrievalModelsConfig(BaseModel):
    planner: str = "gpt-4o-2024-11-20"
    paraphrase: str = "gpt-4o-2024-11-20"
    validator: str = "gpt-4o-2024-11-20"
    none_analyzer: str = "gpt-4o-2024-11-20"
    interpreter: str = "gpt-4o-2024-11-20"
    reranker: str = "gpt-4o-2024-11-20"


class LLMModelsConfig(BaseModel):
    indexing: LLMIndexingModelsConfig = Field(default_factory=LLMIndexingModelsConfig)
    retrieval: LLMRetrievalModelsConfig = Field(
        default_factory=LLMRetrievalModelsConfig
    )


class LLMConfig(BaseModel):
    route: str = "vercel_ai_sdk_python"
    provider_mode: str = "direct"
    openrouter: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "api_key_env": "OPENROUTER_API_KEY"}
    )
    models: LLMModelsConfig = Field(default_factory=LLMModelsConfig)


class SceneExtractionConfig(BaseModel):
    shot_threshold: int = 30
    shot_frame_count: int = 10
    time_interval: int = 1
    time_frame_count: int = 10


class TranscriptConfig(BaseModel):
    method: str = "index_spoken_words"
    engine: str = "gemini"
    language_code: str = ""


class ObjectDetectionConfig(BaseModel):
    provider: str = "rtdetr_v2"
    mode: str = "local"
    backend: str = "rtdetr_v2"
    threshold: float = 0.85
    batch_size: int = 64
    resume_chunk_size: int = 256


class VLMConfig(BaseModel):
    llm_max_images: int = 10
    batch_size: int = 500
    max_concurrent_llm_calls: int = 8
    generate_subplot: bool = True
    subplot_chunk_size: int = 250
    retry_attempts: int = 3
    retry_backoff_sec: int = 1
    temperature: float = 0
    thinking_budget: str = "low"
    prompt_file: Optional[str] = None


class IndexingConfig(BaseModel):
    scene_extraction: SceneExtractionConfig = Field(
        default_factory=SceneExtractionConfig
    )
    transcript: TranscriptConfig = Field(default_factory=TranscriptConfig)
    object_detection: ObjectDetectionConfig = Field(
        default_factory=ObjectDetectionConfig
    )
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    overwrite_existing_indexes: bool = True
    debug_mode: bool = False


class RetrievalConfig(BaseModel):
    page_size: int = 10
    k_variants_per_index: int = 2
    topk_per_variant: int = 30
    score_threshold: float = 0.0
    dynamic_score_percentage: int = 100
    validator_max: int = 40
    validator_batch_size: int = 8
    max_ops_per_batch: int = 4
    recursion_limit: int = 12
    debug_mode: bool = False


class DeepSearchConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @classmethod
    def defaults(cls) -> DeepSearchConfig:
        return cls()

    @classmethod
    def from_file(cls, path: str) -> DeepSearchConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def from_env(cls, prefix: str = "DEEPSEARCH_") -> DeepSearchConfig:
        overrides: Dict[str, Any] = {}
        for key, val in os.environ.items():
            if key.startswith(prefix):
                nested = key[len(prefix) :].lower().split("__")
                cursor = overrides
                for part in nested[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[nested[-1]] = _coerce_env_value(val)
        return cls.model_validate(overrides) if overrides else cls()

    def with_overrides(self, overrides: Dict[str, Any]) -> DeepSearchConfig:
        data = self.model_dump()
        _deep_merge(data, overrides)
        return self.model_validate(data)


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _coerce_env_value(value: str) -> Union[str, bool, int, float]:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
