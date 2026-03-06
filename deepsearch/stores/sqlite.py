from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, Optional

from deepsearch.stores.base import (
    IndexArtifactStore,
    IndexRecordStore,
    MetadataStore,
    SessionStore,
)

_DEFAULT_DB_PATH = "deepsearch.db"


def _get_db_path(path: Optional[str] = None) -> str:
    return path or _DEFAULT_DB_PATH


class _SQLiteBase:
    def __init__(self, db_path: Optional[str] = None):
        self._db_path = _get_db_path(db_path)
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn


class SQLiteSessionStore(_SQLiteBase, SessionStore):
    def __init__(self, db_path: Optional[str] = None):
        super().__init__(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ds_sessions (
                session_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

    def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT INTO ds_sessions (session_id, state, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(session_id)
               DO UPDATE SET state = excluded.state, updated_at = CURRENT_TIMESTAMP""",
            (session_id, json.dumps(state, default=str)),
        )
        conn.commit()

    def load_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = (
            self._conn()
            .execute(
                "SELECT state FROM ds_sessions WHERE session_id = ?", (session_id,)
            )
            .fetchone()
        )
        return json.loads(row["state"]) if row else None

    def delete_state(self, session_id: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM ds_sessions WHERE session_id = ?", (session_id,))
        conn.commit()


class SQLiteMetadataStore(_SQLiteBase, MetadataStore):
    def __init__(self, db_path: Optional[str] = None):
        super().__init__(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ds_collection_metadata (
                collection_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                facet_name TEXT NOT NULL,
                facet_values TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (collection_id, video_id, facet_name)
            )
            """
        )
        conn.commit()

    def save_metadata(
        self, collection_id: str, video_id: str, metadata: Dict[str, Any]
    ) -> None:
        conn = self._conn()
        for facet_name, values in metadata.items():
            vals = values if isinstance(values, list) else [values]
            conn.execute(
                """INSERT INTO ds_collection_metadata (collection_id, video_id, facet_name, facet_values, updated_at)
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(collection_id, video_id, facet_name)
                   DO UPDATE SET facet_values = excluded.facet_values, updated_at = CURRENT_TIMESTAMP""",
                (collection_id, video_id, facet_name, json.dumps(vals)),
            )
        conn.commit()

    def get_collection_metadata(self, collection_id: str) -> Dict[str, Any]:
        rows = (
            self._conn()
            .execute(
                "SELECT facet_name, facet_values FROM ds_collection_metadata WHERE collection_id = ?",
                (collection_id,),
            )
            .fetchall()
        )
        merged: Dict[str, set] = {}
        for row in rows:
            values = json.loads(row["facet_values"])
            merged.setdefault(row["facet_name"], set()).update(values)
        return {k: sorted(v) for k, v in merged.items()}

    def delete_video_metadata(self, collection_id: str, video_id: str) -> None:
        conn = self._conn()
        conn.execute(
            "DELETE FROM ds_collection_metadata WHERE collection_id = ? AND video_id = ?",
            (collection_id, video_id),
        )
        conn.commit()


class SQLiteIndexRecordStore(_SQLiteBase, IndexRecordStore):
    def __init__(self, db_path: Optional[str] = None):
        super().__init__(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ds_index_records (
                collection_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                record TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (collection_id, source_key)
            )
            """
        )
        conn.commit()

    def save_index_record(
        self, collection_id: str, source_key: str, record: Dict[str, Any]
    ) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT INTO ds_index_records (collection_id, source_key, record, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(collection_id, source_key)
               DO UPDATE SET record = excluded.record, updated_at = CURRENT_TIMESTAMP""",
            (collection_id, source_key, json.dumps(record, default=str)),
        )
        conn.commit()

    def load_index_record(
        self, collection_id: str, source_key: str
    ) -> Optional[Dict[str, Any]]:
        row = (
            self._conn()
            .execute(
                "SELECT record FROM ds_index_records WHERE collection_id = ? AND source_key = ?",
                (collection_id, source_key),
            )
            .fetchone()
        )
        return json.loads(row["record"]) if row else None


class SQLiteIndexArtifactStore(_SQLiteBase, IndexArtifactStore):
    def __init__(self, db_path: Optional[str] = None):
        super().__init__(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ds_index_artifacts (
                collection_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                artifact_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (collection_id, video_id, artifact_name)
            )
            """
        )
        conn.commit()

    def save_index_artifact(
        self,
        collection_id: str,
        video_id: str,
        artifact_name: str,
        payload: Dict[str, Any] | list[Any],
    ) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT INTO ds_index_artifacts (collection_id, video_id, artifact_name, payload, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(collection_id, video_id, artifact_name)
               DO UPDATE SET payload = excluded.payload, updated_at = CURRENT_TIMESTAMP""",
            (collection_id, video_id, artifact_name, json.dumps(payload, default=str)),
        )
        conn.commit()

    def load_index_artifact(
        self, collection_id: str, video_id: str, artifact_name: str
    ) -> Optional[Dict[str, Any] | list[Any]]:
        row = (
            self._conn()
            .execute(
                "SELECT payload FROM ds_index_artifacts WHERE collection_id = ? AND video_id = ? AND artifact_name = ?",
                (collection_id, video_id, artifact_name),
            )
            .fetchone()
        )
        return json.loads(row["payload"]) if row else None
