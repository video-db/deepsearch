from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional, Union

import videodb
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from deepsearch.config.schema import DeepSearchConfig
from deepsearch.errors.codes import DS_PIPELINE_STAGE_ERROR, DeepSearchError
from deepsearch.indexing.contracts import IndexManifest
from deepsearch.indexing.pipeline import IndexingPipeline
from deepsearch.providers.detector.base import DetectorProvider
from deepsearch.providers.llm.base import LLMProvider, NodeLLM
from deepsearch.providers.llm.registry import create_llm_provider
from deepsearch.retrieval.contracts import (
    ClipExplain,
    ClipResult,
    ClarificationQuestion,
    PageInfo,
    RetrievalResult,
    UiEvent,
)
from deepsearch.retrieval.graph import build_graph
from deepsearch.retrieval.helpers.prompts import PromptFactory
from deepsearch.retrieval.helpers.schema import History, JoinedShot
from deepsearch.retrieval.state import GraphState
from deepsearch.stores.base import MetadataStore, SessionStore
from deepsearch.stores.base import IndexArtifactStore, IndexRecordStore
from deepsearch.stores.sqlite import (
    SQLiteIndexArtifactStore,
    SQLiteIndexRecordStore,
    SQLiteMetadataStore,
    SQLiteSessionStore,
)
from deepsearch.telemetry.logger import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()


class DeepSearchClient:
    def __init__(
        self,
        config: Optional[Union[DeepSearchConfig, Dict[str, Any], str]] = None,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        collection_id: Optional[str] = None,
        llm_provider: Optional[LLMProvider] = None,
        detector_provider: Optional[DetectorProvider] = None,
        session_store: Optional[SessionStore] = None,
        metadata_store: Optional[MetadataStore] = None,
        index_record_store: Optional[IndexRecordStore] = None,
        index_artifact_store: Optional[IndexArtifactStore] = None,
    ):
        self.config = self._resolve_config(config)
        debug_env = os.getenv("DEEPSEARCH_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._retrieval_debug_mode = debug_env or self.config.retrieval.debug_mode
        self._indexing_debug_mode = debug_env or self.config.indexing.debug_mode
        self._debug_mode = self._retrieval_debug_mode or self._indexing_debug_mode
        setup_logging(level="DEBUG" if self._debug_mode else "INFO")
        self._api_key = api_key or os.getenv("VIDEO_DB_API_KEY")
        self._base_url = base_url or os.getenv("VIDEO_DB_BASE_URL")
        if not self._api_key:
            raise ValueError("VIDEO_DB_API_KEY is required")
        connect_kwargs = {"api_key": self._api_key}
        if self._base_url:
            connect_kwargs["base_url"] = self._base_url
        self._conn = videodb.connect(**connect_kwargs)
        self._default_collection_id = collection_id
        self._llm_provider = llm_provider or create_llm_provider(self.config.llm)
        self._detector_provider = detector_provider
        db_path = os.getenv("DEEPSEARCH_DB_PATH")
        self._session_store = session_store or SQLiteSessionStore(db_path)
        self._metadata_store = metadata_store or SQLiteMetadataStore(db_path)
        self._index_record_store = index_record_store or SQLiteIndexRecordStore(db_path)
        self._index_artifact_store = index_artifact_store or SQLiteIndexArtifactStore(
            db_path
        )
        self._graph = build_graph()

    def index_video(
        self,
        video_url: Optional[str] = None,
        *,
        media_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        name: Optional[str] = None,
        on_event=None,
        force_reindex: bool = False,
    ) -> IndexManifest:
        collection = self._get_collection(collection_id)
        source_key = self._make_index_source_key(video_url=video_url, media_id=media_id)
        existing = self._index_record_store.load_index_record(collection.id, source_key)
        if existing and not force_reindex:
            if existing.get("status") == "completed" and existing.get("manifest"):
                logger.info(
                    "Skipping indexing and returning stored manifest source_key=%s",
                    source_key,
                )
                return IndexManifest.model_validate(existing["manifest"])
            if video_url and existing.get("video_id"):
                logger.info(
                    "Resuming indexing from existing uploaded media_id=%s source_key=%s",
                    existing["video_id"],
                    source_key,
                )
                media_id = existing["video_id"]
                video_url = None
        pipeline = IndexingPipeline(
            config=self.config,
            metadata_store=self._metadata_store,
            detector_provider=self._detector_provider,
            on_event=on_event,
            index_record_store=self._index_record_store,
            index_artifact_store=self._index_artifact_store,
            source_key=source_key,
        )
        return pipeline.run(
            collection=collection,
            collection_id=collection.id,
            video_url=video_url,
            media_id=media_id,
            name=name,
        )

    def _make_index_source_key(
        self, *, video_url: Optional[str], media_id: Optional[str]
    ) -> str:
        return f"media_id:{media_id}" if media_id else f"video_url:{video_url}"

    def start_session(
        self,
        collection_id: Optional[str] = None,
        video_id: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> "DeepSearchSession":
        collection = self._get_collection(collection_id)
        session_id = str(uuid.uuid4())
        return DeepSearchSession(
            session_id=session_id,
            client=self,
            collection_id=collection.id,
            video_id=video_id,
            session_page_size=page_size,
        )

    def resume_session(self, session_id: str) -> "DeepSearchSession":
        saved = self._load_session(session_id)
        if not saved:
            raise ValueError(f"Unknown session_id: {session_id}")
        return DeepSearchSession(
            session_id=session_id,
            client=self,
            collection_id=saved["collection_id"],
            video_id=saved.get("video_id"),
            session_page_size=saved.get("page_size"),
            persisted_state=saved,
        )

    def _resolve_config(
        self, config: Optional[Union[DeepSearchConfig, Dict[str, Any], str]]
    ) -> DeepSearchConfig:
        if config is None:
            return DeepSearchConfig.defaults()
        if isinstance(config, DeepSearchConfig):
            return config
        if isinstance(config, dict):
            return DeepSearchConfig.model_validate(config)
        if isinstance(config, str):
            return DeepSearchConfig.from_file(config)
        raise TypeError("config must be DeepSearchConfig, dict, path, or None")

    def _get_collection(self, collection_id: Optional[str]):
        cid = collection_id or self._default_collection_id
        return self._conn.get_collection(cid) if cid else self._conn.get_collection()

    def _make_llms(self) -> Dict[str, NodeLLM]:
        retrieval_models = self.config.llm.models.retrieval
        return {
            "planner": NodeLLM(
                self._llm_provider, retrieval_models.planner, task="planner"
            ),
            "paraphrase": NodeLLM(
                self._llm_provider, retrieval_models.paraphrase, task="paraphrase"
            ),
            "validator": NodeLLM(
                self._llm_provider, retrieval_models.validator, task="validator"
            ),
            "none_analyzer": NodeLLM(
                self._llm_provider, retrieval_models.none_analyzer, task="none_analyzer"
            ),
            "interpreter": NodeLLM(
                self._llm_provider, retrieval_models.interpreter, task="interpreter"
            ),
            "reranker": NodeLLM(
                self._llm_provider, retrieval_models.reranker, task="reranker"
            ),
        }

    def _invoke_graph(self, state: GraphState) -> dict:
        try:
            return self._graph.invoke(
                state, {"recursion_limit": self.config.retrieval.recursion_limit}
            )
        except GraphRecursionError as exc:
            logger.error("Graph recursion limit reached")
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR,
                "Graph recursion limit reached",
                stage_or_node="retrieval",
                retryable=True,
            ) from exc

    def _save_session(self, session_id: str, state: Dict[str, Any]) -> None:
        self._session_store.save_state(session_id, state)

    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._session_store.load_state(session_id)

    def _build_runtime_state(self, persisted: Dict[str, Any]) -> GraphState:
        collection = self._get_collection(persisted["collection_id"])
        unique_metadata = (
            self._metadata_store.get_collection_metadata(collection.id)
            if self._metadata_store
            else {}
        )
        if self._retrieval_debug_mode:
            logger.debug(
                "Runtime state metadata collection_id=%s unique_metadata=%s",
                collection.id,
                unique_metadata,
            )
        prompts = PromptFactory(
            unique_metadata,
            debug_mode=self._retrieval_debug_mode,
        )
        history = persisted.get("history") or {}
        plan = persisted.get("plan")
        return GraphState.model_validate(
            {
                **persisted,
                "collection": collection,
                "collection_id": collection.id,
                "prompts": prompts,
                "llms": self._make_llms(),
                "history": history,
                "plan": plan,
            }
        )


class DeepSearchSession:
    def __init__(
        self,
        session_id: str,
        client: DeepSearchClient,
        collection_id: str,
        video_id: Optional[str] = None,
        session_page_size: Optional[int] = None,
        persisted_state: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self._client = client
        self.collection_id = collection_id
        self.video_id = video_id
        self._session_page_size = session_page_size
        self._persisted = persisted_state
        self._video_cache: Dict[str, Any] = {}
        self._stream_cache: Dict[str, str] = {}

    def search(self, query: str, *, page_size: Optional[int] = None) -> RetrievalResult:
        state = GraphState(
            session_id=self.session_id,
            collection_id=self.collection_id,
            collection=self._client._get_collection(self.collection_id),
            video_id=self.video_id,
            cfg=self._client.config.retrieval,
            llms=self._client._make_llms(),
            prompts=PromptFactory(
                self._client._metadata_store.get_collection_metadata(self.collection_id)
                if self._client._metadata_store
                else {},
                debug_mode=self._client._retrieval_debug_mode,
            ),
            main_query=query,
            history=History(),
            page_size=self._resolved_page_size(page_size),
        )
        new_state = self._client._invoke_graph(state)
        persisted = self._persist_state(
            new_state, self._resolved_page_size(page_size), query
        )
        self._persisted = persisted
        return self._build_result(persisted, page_size=page_size)

    def followup(
        self,
        text: Optional[str] = None,
        *,
        ui_event: Optional[Union[UiEvent, Dict[str, Any]]] = None,
        page_size: Optional[int] = None,
    ) -> RetrievalResult:
        if text is None and ui_event is None:
            raise ValueError("At least one of text or ui_event is required")
        persisted = self._require_state()
        resolved_page_size = self._resolved_page_size(page_size, persisted)
        event = (
            UiEvent.model_validate(ui_event)
            if ui_event is not None and not isinstance(ui_event, UiEvent)
            else ui_event
        )
        if event and event.type == "show_more":
            persisted["page_size"] = resolved_page_size
            self._persisted = persisted
            self._client._save_session(self.session_id, persisted)
            return self._build_result(
                persisted, page_size=resolved_page_size, advance=True
            )

        runtime_state = self._client._build_runtime_state(
            {
                **persisted,
                "user_text": text,
                "ui_event": event.model_dump() if event else None,
                "page_size": resolved_page_size,
            }
        )
        new_state = self._client._invoke_graph(runtime_state)
        updated = self._persist_state(
            new_state, resolved_page_size, persisted.get("main_query", "")
        )
        self._persisted = updated
        return self._build_result(updated, page_size=resolved_page_size)

    @property
    def result(self) -> RetrievalResult:
        return self._build_result(self._require_state())

    def _require_state(self) -> Dict[str, Any]:
        state = self._persisted or self._client._load_session(self.session_id)
        if not state:
            raise ValueError(f"No persisted state found for session {self.session_id}")
        return state

    def _resolved_page_size(
        self, page_size: Optional[int], persisted: Optional[Dict[str, Any]] = None
    ) -> int:
        if page_size is not None:
            return page_size
        if persisted and persisted.get("page_size"):
            return int(persisted["page_size"])
        if self._session_page_size is not None:
            return self._session_page_size
        return self._client.config.retrieval.page_size

    def _persist_state(
        self, state: Dict[str, Any], page_size: int, main_query: str
    ) -> Dict[str, Any]:
        persisted = {
            "session_id": self.session_id,
            "collection_id": self.collection_id,
            "video_id": self.video_id,
            "main_query": main_query or state.get("main_query", ""),
            "plan": state.get("plan").model_dump()
            if hasattr(state.get("plan"), "model_dump")
            else state.get("plan"),
            "history": state.get("history").model_dump()
            if hasattr(state.get("history"), "model_dump")
            else state.get("history") or {},
            "paused_for": state.get("paused_for"),
            "clarify_question": state.get("clarify_question"),
            "feedback": state.get("feedback"),
            "validator_verdicts": state.get("validator_verdicts") or {},
            "page_cursor": 0,
            "page_size": page_size,
            "ranked_shots": [
                shot.model_dump() if hasattr(shot, "model_dump") else shot
                for shot in (state.get("ranked_shots") or [])
            ],
        }
        persisted["ranked_cache"] = [
            clip.model_dump() for clip in self._build_ranked_cache(persisted)
        ]
        persisted["last_preview"] = []
        self._client._save_session(self.session_id, persisted)
        return persisted

    def _build_ranked_cache(self, persisted: Dict[str, Any]) -> list[ClipResult]:
        clips = []
        ranked_shots = persisted.get("ranked_shots") or []
        verdicts = persisted.get("validator_verdicts") or {}
        for i, raw in enumerate(ranked_shots, start=1):
            shot = raw if isinstance(raw, dict) else raw.model_dump()
            score = float((shot.get("primary") or {}).get("search_score") or 0.0)
            key = f"{shot.get('video_id', '')}:{shot.get('start', 0)}:{shot.get('end', 0)}"
            clips.append(
                ClipResult(
                    video_id=shot.get("video_id", ""),
                    start=float(shot.get("start", 0)),
                    end=float(shot.get("end", 0)),
                    rank=i,
                    score=score,
                    validator_status=verdicts.get(key, "pass"),
                    stream_url=self._playable_ref(shot),
                    explain=ClipExplain(
                        primary_subquery=str(
                            (shot.get("primary") or {}).get("subquery_id") or ""
                        ),
                        primary_index=str(
                            (shot.get("primary") or {}).get("index") or ""
                        ),
                        support_subqueries=list(shot.get("support_subqueries") or []),
                    ),
                )
            )
        return clips

    def _playable_ref(self, shot: Dict[str, Any]) -> Optional[str]:
        try:
            video_id = shot.get("video_id")
            if not video_id:
                return None
            start = float(shot.get("start", 0) or 0)
            end = float(shot.get("end", 0) or 0)
            cache_key = f"{video_id}:{start:.6f}:{end:.6f}"
            if cache_key in self._stream_cache:
                return self._stream_cache[cache_key]
            if video_id not in self._video_cache:
                self._video_cache[video_id] = self._client._get_collection(
                    self.collection_id
                ).get_video(video_id)
            video = self._video_cache[video_id]
            stream_url = video.generate_stream(timeline=[(start, end)])
            if stream_url:
                self._stream_cache[cache_key] = stream_url
            return stream_url
        except Exception:
            return None

    def _build_result(
        self,
        persisted: Dict[str, Any],
        *,
        page_size: Optional[int] = None,
        advance: bool = False,
    ) -> RetrievalResult:
        page_size = self._resolved_page_size(page_size, persisted)
        ranked_cache = [
            ClipResult.model_validate(item)
            for item in persisted.get("ranked_cache", [])
        ]
        cursor = int(persisted.get("page_cursor", 0))
        if advance:
            cursor = min(cursor + page_size, len(ranked_cache))
            persisted["page_cursor"] = cursor
            self._client._save_session(self.session_id, persisted)
        current_page = ranked_cache[cursor : cursor + page_size]
        has_more = cursor + len(current_page) < len(ranked_cache)
        next_cursor = cursor + len(current_page) if has_more else None
        persisted["last_preview"] = [clip.model_dump() for clip in current_page]
        self._client._save_session(self.session_id, persisted)
        waiting_for = "none"
        clarification = None
        if persisted.get("paused_for") == "clarify_pause":
            waiting_for = "clarification"
            clarification = ClarificationQuestion.model_validate(
                persisted.get("clarify_question") or {}
            )
        elif persisted.get("paused_for") == "preview_pause":
            waiting_for = "user_input"
        debug = None
        if advance and not current_page and not has_more:
            debug = {"status_hint": "no_more_results"}
        return RetrievalResult(
            session_id=self.session_id,
            clips=[
                clip.model_copy(update={"rank": i + 1})
                for i, clip in enumerate(current_page)
            ],
            waiting_for=waiting_for,
            clarification=clarification,
            page=PageInfo(
                page_size=page_size,
                cursor=cursor,
                next_cursor=next_cursor,
                has_more=has_more,
            ),
            debug=debug,
        )
