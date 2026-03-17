from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SessionStore(ABC):
    @abstractmethod
    def save_state(self, session_id: str, state: Dict[str, Any]) -> None: ...

    @abstractmethod
    def load_state(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def delete_state(self, session_id: str) -> None: ...


class MetadataStore(ABC):
    """Stores per-collection unique facet values extracted during indexing."""

    @abstractmethod
    def save_metadata(
        self, collection_id: str, video_id: str, metadata: Dict[str, Any]
    ) -> None: ...

    @abstractmethod
    def get_collection_metadata(self, collection_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    def delete_video_metadata(self, collection_id: str, video_id: str) -> None: ...


class IndexRecordStore(ABC):
    @abstractmethod
    def save_index_record(
        self, collection_id: str, source_key: str, record: Dict[str, Any]
    ) -> None: ...

    @abstractmethod
    def load_index_record(
        self, collection_id: str, source_key: str
    ) -> Optional[Dict[str, Any]]: ...


class SearchClient(ABC):
    """Abstraction for vector search, allowing server-side implementations
    to bypass the VideoDB SDK round-trip."""

    num_db_calls: int

    @abstractmethod
    def search(
        self,
        coll,
        index_list: list,
        q: str,
        metadata_filters: Dict[str, List[str]],
        topk: int,
        sid: Optional[str],
        score_threshold: float,
        dynamic_score_percentage: int,
    ) -> List: ...


class IndexArtifactStore(ABC):
    @abstractmethod
    def save_index_artifact(
        self,
        collection_id: str,
        video_id: str,
        artifact_name: str,
        payload: Dict[str, Any] | List[Any],
    ) -> None: ...

    @abstractmethod
    def load_index_artifact(
        self, collection_id: str, video_id: str, artifact_name: str
    ) -> Optional[Dict[str, Any] | List[Any]]: ...
