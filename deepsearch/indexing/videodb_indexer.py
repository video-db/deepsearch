from __future__ import annotations

import functools
import json
import logging
import operator
import os
from typing import Any, Dict, List, Optional, Tuple

from videodb.scene import Scene

logger = logging.getLogger(__name__)


def _flatten_dotpath(data: Dict[str, Any], path: str) -> Any:
    try:
        return functools.reduce(operator.getitem, path.split("."), data)
    except (KeyError, TypeError):
        return None


class VideoDBIndexer:
    BASE_REQUIRED_INDEXES = {
        "location",
        "scene_description",
        "transcript",
        "topic",
        "object_description",
    }

    SUMMARY_INDEXES = {
        "subplot_summary",
        "final_summary",
    }

    REQUIRED_INDEXES = BASE_REQUIRED_INDEXES | SUMMARY_INDEXES

    INDEX_FIELDS = {
        "action": "action",
        "location": "location",
        "scene_description": "scene_description",
        "object_description": "object_description",
        "transcript": "transcript",
        "topic": "topic.description",
    }

    METADATA_FIELDS = {
        "emotion.emotion_type": "emotion",
        "emotion.type": "emotion",
        "objects": "objects",
        "detected_objects": "objects",
        "shot_type": "shot_type",
    }

    def __init__(self, video, config: Optional[Dict[str, Any]] = None):
        self.video = video
        self.config = config or {}
        self.created_indexes = {}
        self.replaced_indexes: List[Dict[str, str]] = []
        self.failed_indexes: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def required_indexes(cls, include_summary: bool) -> set[str]:
        return (
            cls.BASE_REQUIRED_INDEXES | cls.SUMMARY_INDEXES
            if include_summary
            else set(cls.BASE_REQUIRED_INDEXES)
        )

    def index_from_files(
        self, compiled_scenes_path: str, subplot_summary_path: Optional[str] = None
    ) -> Dict[str, Any]:
        with open(compiled_scenes_path, "r", encoding="utf-8") as f:
            compiled_scenes = json.load(f)
        subplot_data = None
        if subplot_summary_path and os.path.exists(subplot_summary_path):
            with open(subplot_summary_path, "r", encoding="utf-8") as f:
                subplot_data = json.load(f)
        return self.create_indexes(compiled_scenes, subplot_data)

    def create_indexes(
        self, compiled_scenes: List[Dict], subplot_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        if not compiled_scenes:
            return {"indexes": {}, "total_scenes": 0, "replaced_indexes": []}

        video_id = self.video.id
        video_length = max((s.get("end", 0) for s in compiled_scenes), default=0)

        self.created_indexes = {}
        self.replaced_indexes = []
        self.failed_indexes = {}
        buckets = {name: [] for name in self.INDEX_FIELDS}
        buckets["subplot_summary"] = []
        buckets["final_summary"] = []

        for scene_dict in compiled_scenes:
            metadata = self._build_metadata(scene_dict)
            for index_name, field_path in self.INDEX_FIELDS.items():
                description = self._normalize_index_value(
                    scene_dict, index_name, field_path
                )
                if description:
                    buckets[index_name].append(
                        Scene(
                            video_id=video_id,
                            start=scene_dict.get("start", 0),
                            end=scene_dict.get("end", 0),
                            description=str(description),
                            metadata=metadata,
                        )
                    )

        if subplot_data:
            for subplot in subplot_data.get("subplots", []):
                buckets["subplot_summary"].append(
                    Scene(
                        video_id=video_id,
                        start=subplot.get("start", 0),
                        end=subplot.get("end", 0),
                        description=subplot.get("summary", ""),
                        metadata={},
                    )
                )
            final_summary = subplot_data.get("final_summary")
            if final_summary:
                buckets["final_summary"].append(
                    Scene(
                        video_id=video_id,
                        start=0,
                        end=video_length,
                        description=final_summary,
                        metadata={},
                    )
                )

        for index_name, scenes in buckets.items():
            if not scenes:
                continue
            self._create_or_update_index(index_name, scenes)

        if self.failed_indexes and self.config.get("retry_failed_indexes", True):
            logger.warning(
                "Retrying failed indexes once: %s",
                sorted(self.failed_indexes.keys()),
            )
            for index_name in list(self.failed_indexes.keys()):
                scenes = buckets.get(index_name, [])
                if not scenes:
                    continue
                self._create_or_update_index(index_name, scenes, retry=True)

        return {
            "indexes": self.created_indexes,
            "total_scenes": len(compiled_scenes),
            "video_length": video_length,
            "replaced_indexes": self.replaced_indexes,
            "failed_indexes": self.failed_indexes,
            "required_indexes_present": sorted(
                self.REQUIRED_INDEXES.intersection(self.created_indexes)
            ),
        }

    def _create_or_update_index(
        self,
        index_name: str,
        scenes: List[Scene],
        retry: bool = False,
    ) -> None:
        try:
            replaced = self._replace_existing_index(index_name)
            index_id = self.video.index_scenes(scenes=scenes, name=index_name)
            self.created_indexes[index_name] = {
                "index_id": index_id,
                "scene_count": len(scenes),
            }
            self.failed_indexes.pop(index_name, None)
            if replaced:
                self.replaced_indexes.append(
                    {
                        "index_name": index_name,
                        "old_index_id": replaced,
                        "new_index_id": index_id,
                    }
                )
            logger.info(
                "%s index '%s' with %s scenes",
                "Recreated" if retry else "Created",
                index_name,
                len(scenes),
            )
        except Exception as e:
            logger.error(
                "%s to create index '%s': %s",
                "Retry failed" if retry else "Failed",
                index_name,
                e,
            )
            error_record = {
                "error": str(e),
                "scene_count": len(scenes),
                "retry_attempted": retry,
            }
            self.created_indexes[index_name] = error_record
            self.failed_indexes[index_name] = error_record

    def _normalize_index_value(
        self, scene_dict: Dict[str, Any], index_name: str, field_path: str
    ) -> Any:
        if index_name == "topic":
            topic = scene_dict.get("topic")
            if isinstance(topic, str):
                return topic
        return _flatten_dotpath(scene_dict, field_path)

    def _replace_existing_index(self, index_name: str) -> Optional[str]:
        if not self.config.get("overwrite_existing_indexes", True):
            return None
        existing = self.video.list_scene_index() or []
        for item in existing:
            name = item.get("name") if isinstance(item, dict) else None
            scene_index_id = (
                item.get("scene_index_id") if isinstance(item, dict) else None
            )
            if name == index_name and scene_index_id:
                self.video.delete_scene_index(scene_index_id)
                return scene_index_id
        return None

    def _build_metadata(self, scene_dict):
        metadata = {}
        for src_path, target_key in self.METADATA_FIELDS.items():
            value = _flatten_dotpath(scene_dict, src_path)
            if value is not None:
                metadata[target_key] = value
        return metadata
