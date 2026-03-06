from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from deepsearch.stores.base import (
    SessionStore,
    MetadataStore,
    IndexRecordStore,
    IndexArtifactStore,
)


class InMemorySessionStore(SessionStore):
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        self._store[session_id] = copy.deepcopy(state)

    def load_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self._store.get(session_id)
        return copy.deepcopy(data) if data else None

    def delete_state(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class InMemoryMetadataStore(MetadataStore):
    def __init__(self):
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def save_metadata(
        self, collection_id: str, video_id: str, metadata: Dict[str, Any]
    ) -> None:
        self._store.setdefault(collection_id, {})[video_id] = copy.deepcopy(metadata)

    def get_collection_metadata(self, collection_id: str) -> Dict[str, Any]:
        videos = self._store.get(collection_id, {})
        merged: Dict[str, set] = {}
        for vid_meta in videos.values():
            for k, v in vid_meta.items():
                if isinstance(v, list):
                    merged.setdefault(k, set()).update(v)
        return {k: sorted(v) for k, v in merged.items()}

    def delete_video_metadata(self, collection_id: str, video_id: str) -> None:
        if collection_id in self._store:
            self._store[collection_id].pop(video_id, None)


class InMemoryIndexRecordStore(IndexRecordStore):
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def save_index_record(
        self, collection_id: str, source_key: str, record: Dict[str, Any]
    ) -> None:
        self._store[f"{collection_id}:{source_key}"] = copy.deepcopy(record)

    def load_index_record(
        self, collection_id: str, source_key: str
    ) -> Optional[Dict[str, Any]]:
        data = self._store.get(f"{collection_id}:{source_key}")
        return copy.deepcopy(data) if data else None


class InMemoryIndexArtifactStore(IndexArtifactStore):
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def save_index_artifact(
        self,
        collection_id: str,
        video_id: str,
        artifact_name: str,
        payload: Dict[str, Any] | List[Any],
    ) -> None:
        self._store[f"{collection_id}:{video_id}:{artifact_name}"] = copy.deepcopy(
            payload
        )

    def load_index_artifact(
        self, collection_id: str, video_id: str, artifact_name: str
    ) -> Optional[Dict[str, Any] | List[Any]]:
        data = self._store.get(f"{collection_id}:{video_id}:{artifact_name}")
        return copy.deepcopy(data) if data is not None else None
