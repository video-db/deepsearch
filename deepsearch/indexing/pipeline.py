from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm
from videodb import SceneExtractionType

from deepsearch.config.schema import DeepSearchConfig
from deepsearch.errors.codes import (
    DS_MISSING_INDEX_ERROR,
    DS_PIPELINE_STAGE_ERROR,
    DeepSearchError,
)
from deepsearch.indexing.contracts import (
    IndexArtifact,
    IndexEvent,
    IndexManifest,
    IndexStageStatus,
    IndexStats,
    ReplacedIndexRef,
    StageTiming,
)
from deepsearch.indexing.subplot_summarizer import SubplotSummarizer
from deepsearch.indexing.videodb_indexer import VideoDBIndexer
from deepsearch.indexing.vlm_extractor import VLMExtractor
from deepsearch.providers.detector.rtdetr import RTDetrDetector
from deepsearch.stores.base import IndexArtifactStore, IndexRecordStore, MetadataStore

logger = logging.getLogger(__name__)


class IndexingPipeline:
    STAGES = [
        "extract",
        "transcript",
        "detect",
        "enrich",
        "summarize",
        "write_indexes",
        "manifest",
    ]

    def __init__(
        self,
        config: DeepSearchConfig,
        metadata_store: Optional[MetadataStore] = None,
        detector_provider=None,
        on_event: Optional[Callable[[IndexEvent], None]] = None,
        index_record_store: Optional[IndexRecordStore] = None,
        index_artifact_store: Optional[IndexArtifactStore] = None,
        source_key: Optional[str] = None,
    ):
        self.config = config
        self.idx_cfg = config.indexing
        self.llm_cfg = config.llm
        self.metadata_store = metadata_store
        self.detector_provider = detector_provider
        self._on_event = on_event
        self.index_record_store = index_record_store
        self.index_artifact_store = index_artifact_store
        self.source_key = source_key
        self._stage_statuses = {
            stage: IndexStageStatus(stage=stage) for stage in self.STAGES
        }
        self._stage_start_times: Dict[str, float] = {}
        self._warnings: List[str] = []
        self._stats = IndexStats()

    def run(
        self,
        *,
        collection: Any,
        collection_id: str,
        video_url: Optional[str] = None,
        media_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> IndexManifest:
        if bool(video_url) == bool(media_id):
            raise ValueError("Exactly one of video_url or media_id is required")

        manifest_id = f"idx-{uuid.uuid4()}"
        work_dir = tempfile.mkdtemp(prefix="deepsearch_idx_")
        ctx: Dict[str, Any] = {
            "manifest_id": manifest_id,
            "collection": collection,
            "collection_id": collection_id,
            "video_url": video_url,
            "media_id": media_id,
            "name": name,
            "work_dir": work_dir,
            "source_key": self.source_key,
        }
        logger.info(
            "Starting indexing manifest_id=%s collection_id=%s input=%s llm_route=%s scene_model=%s summary_model=%s detect_mode=%s detect_provider=%s work_dir=%s",
            manifest_id,
            collection_id,
            "media_id" if media_id else "video_url",
            self.llm_cfg.route,
            self.llm_cfg.models.indexing.scene_enrichment,
            self.llm_cfg.models.indexing.final_summary,
            self.idx_cfg.object_detection.mode,
            self.idx_cfg.object_detection.provider,
            work_dir,
        )

        try:
            ctx = self._run_extract_stage(ctx)
            self._write_checkpoint(ctx, current_stage="extract")
            self._maybe_restore_artifacts_for_reindex(ctx)
            if ctx.get("_restored_for_reindex"):
                logger.info(
                    "Using persisted artifacts for video_id=%s and skipping transcript/detect/enrich/summarize stages",
                    ctx.get("video_id"),
                )
                self._emit("transcript", "completed", "skipped: restored artifacts")
                self._emit("detect", "completed", "skipped: restored artifacts")
                self._emit("enrich", "completed", "loaded compiled_scenes from DB")
                self._emit("summarize", "completed", "loaded subplot_summary from DB")
            else:
                ctx = self._run_transcript_stage(ctx)
                self._write_checkpoint(ctx, current_stage="transcript")
                ctx = self._run_detect_stage(ctx)
                self._write_checkpoint(ctx, current_stage="detect")
                ctx = self._run_enrich_stage(ctx)
                self._write_checkpoint(ctx, current_stage="enrich")
                ctx = self._run_summarize_stage(ctx)
                self._write_checkpoint(ctx, current_stage="summarize")
            ctx = self._run_write_indexes_stage(ctx)
            self._write_checkpoint(ctx, current_stage="write_indexes")
            return self._run_manifest_stage(ctx)
        except DeepSearchError as exc:
            self._attach_resume_details(exc, ctx)
            self._write_checkpoint(
                ctx,
                current_stage=exc.stage_or_node or "failed",
                failed=True,
                error=str(exc),
            )
            self._save_index_record(ctx, status="failed", error=str(exc))
            raise
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            ds_exc = DeepSearchError(
                code=DS_PIPELINE_STAGE_ERROR,
                message=str(exc),
                stage_or_node="indexing_pipeline",
                retryable=False,
            )
            self._attach_resume_details(ds_exc, ctx)
            self._write_checkpoint(
                ctx, current_stage="indexing_pipeline", failed=True, error=str(exc)
            )
            self._save_index_record(ctx, status="failed", error=str(exc))
            raise ds_exc from exc

    def _run_extract_stage(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        stage = "extract"
        self._emit(stage, "started", "Resolving input and extracting scenes")
        try:
            video = self._resolve_video(
                ctx["collection"],
                video_url=ctx.get("video_url"),
                media_id=ctx.get("media_id"),
                name=ctx.get("name"),
            )
            shot_scene_id, time_scene_id = self._extract_scenes(video)
            ctx.update(
                {
                    "video": video,
                    "video_id": video.id,
                    "shot_scene_id": shot_scene_id,
                    "time_scene_id": time_scene_id,
                }
            )
            self._emit(
                stage,
                "completed",
                f"Extracted shot/time scene collections for {video.id}",
            )
            return ctx
        except Exception as exc:
            self._emit(stage, "failed", str(exc))
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR, str(exc), stage_or_node=stage, retryable=False
            ) from exc

    def _run_transcript_stage(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        stage = "transcript"
        self._emit(stage, "started", "Indexing spoken words and fetching transcript")
        try:
            self._index_transcript(ctx["video"])
            transcript = self._get_transcript_for_vlm(ctx["video"])
            ctx["transcript"] = transcript
            self._emit(
                stage, "completed", f"Fetched {len(transcript)} transcript segments"
            )
            return ctx
        except Exception as exc:
            self._emit(stage, "failed", str(exc))
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR, str(exc), stage_or_node=stage, retryable=False
            ) from exc

    def _run_detect_stage(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        stage = "detect"
        self._emit(stage, "started", "Running object detection")
        try:
            provider = self.detector_provider or self._build_detector_provider()
            vision_metadata = self._run_object_detection(
                ctx["collection_id"],
                ctx["video"],
                ctx["time_scene_id"],
                ctx["work_dir"],
                provider,
            )
            ctx["vision_metadata"] = vision_metadata
            self._save_artifact(ctx, "vision_metadata", vision_metadata)
            message = (
                "skipped"
                if provider is None
                else f"Processed {len(vision_metadata)} frame windows"
            )
            self._emit(stage, "completed", message)
            return ctx
        except Exception as exc:
            self._emit(stage, "failed", str(exc))
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR, str(exc), stage_or_node=stage, retryable=False
            ) from exc

    def _run_enrich_stage(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        stage = "enrich"
        self._emit(stage, "started", "Running multimodal scene enrichment")
        try:
            shot_scenes = ctx["video"].get_scene_collection(ctx["shot_scene_id"]).scenes
            frame_scenes = (
                ctx["video"].get_scene_collection(ctx["time_scene_id"]).scenes
            )
            vlm_result = self._run_vlm(
                ctx["transcript"],
                ctx["vision_metadata"],
                shot_scenes,
                frame_scenes,
                ctx["work_dir"],
            )
            compiled_scenes = self._load_compiled_scenes(vlm_result["compiled_path"])
            ctx["vlm_result"] = vlm_result
            ctx["compiled_scenes"] = compiled_scenes
            self._save_artifact(ctx, "compiled_scenes", compiled_scenes)
            self._stats.token_usage[stage] = int(
                vlm_result.get("prompt_tokens", 0)
            ) + int(vlm_result.get("completion_tokens", 0))
            self._emit(
                stage,
                "completed",
                f"Enriched {vlm_result.get('scenes_processed', 0)} scenes",
            )
            return ctx
        except Exception as exc:
            self._emit(stage, "failed", str(exc))
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR, str(exc), stage_or_node=stage, retryable=False
            ) from exc

    def _run_summarize_stage(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        stage = "summarize"
        self._emit(stage, "started", "Generating subplot and final summaries")
        try:
            subplot_result = None
            if self.idx_cfg.vlm.generate_subplot:
                subplot_result = self._run_subplot_summary(
                    ctx["compiled_scenes"], ctx["work_dir"]
                )
                self._stats.token_usage[stage] = int(
                    subplot_result.get("prompt_tokens", 0)
                ) + int(subplot_result.get("completion_tokens", 0))
                self._emit(
                    stage,
                    "completed",
                    f"Generated {subplot_result.get('subplot_count', 0)} subplot segments",
                )
            else:
                self._emit(stage, "completed", "skipped")
            ctx["subplot_result"] = subplot_result
            if subplot_result and subplot_result.get("subplot_path"):
                with open(subplot_result["subplot_path"], encoding="utf-8") as handle:
                    subplot_data = json.load(handle)
                ctx["subplot_data"] = subplot_data
                self._save_artifact(ctx, "subplot_summary", subplot_data)
            return ctx
        except Exception as exc:
            self._emit(stage, "failed", str(exc))
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR, str(exc), stage_or_node=stage, retryable=False
            ) from exc

    def _run_write_indexes_stage(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        stage = "write_indexes"
        self._emit(stage, "started", "Writing semantic indexes to VideoDB")
        try:
            index_result = self._create_indexes(
                ctx["video"],
                ctx.get("compiled_scenes") or [],
                ctx.get("subplot_data"),
            )
            ctx["index_result"] = index_result
            failed_indexes = index_result.get("failed_indexes", {})
            if failed_indexes:
                self._save_artifact(ctx, "failed_indexes", failed_indexes)
                warning = (
                    "Index creation failed for "
                    + ", ".join(sorted(failed_indexes.keys()))
                    + ". Re-run indexing to retry failed indexes."
                )
                self._warnings.append(warning)
                self._save_index_record(
                    ctx,
                    status="index_partial_failed",
                    error=warning,
                )
            if self.metadata_store:
                self._extract_and_store_metadata(
                    ctx["compiled_scenes"], ctx["collection_id"], ctx["video_id"]
                )
            self._emit(
                stage,
                "completed",
                f"Created {len(index_result.get('indexes', {}))} indexes",
            )
            return ctx
        except Exception as exc:
            self._emit(stage, "failed", str(exc))
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR, str(exc), stage_or_node=stage, retryable=False
            ) from exc

    def _run_manifest_stage(self, ctx: Dict[str, Any]) -> IndexManifest:
        stage = "manifest"
        self._emit(
            stage, "started", "Validating required indexes and building manifest"
        )
        try:
            indexes = ctx["index_result"].get("indexes", {})
            failed_indexes = ctx["index_result"].get("failed_indexes", {})
            missing = sorted(VideoDBIndexer.REQUIRED_INDEXES.difference(indexes))
            missing_errors = sorted(
                name
                for name in VideoDBIndexer.REQUIRED_INDEXES
                if "error" in indexes.get(name, {})
            )
            if missing or missing_errors:
                details = {
                    "missing_indexes": missing,
                    "errored_indexes": missing_errors,
                    "failed_indexes": failed_indexes,
                }
                self._emit(stage, "failed", json.dumps(details, ensure_ascii=False))
                raise DeepSearchError(
                    code=DS_MISSING_INDEX_ERROR,
                    message="Required indexes missing from manifest",
                    stage_or_node=stage,
                    retryable=False,
                    details=details,
                )

            artifacts = {
                name: IndexArtifact(
                    index_name=name,
                    index_id=value["index_id"],
                    scene_count=int(value.get("scene_count", 0)),
                )
                for name, value in indexes.items()
                if value.get("index_id")
            }
            self._stats.total_scenes = int(ctx["index_result"].get("total_scenes", 0))
            self._stats.replaced_indexes = [
                ReplacedIndexRef(**item)
                for item in ctx["index_result"].get("replaced_indexes", [])
            ]
            manifest = IndexManifest(
                manifest_id=ctx["manifest_id"],
                collection_id=ctx["collection_id"],
                video_id=ctx["video_id"],
                indexes=artifacts,
                stats=self._stats,
                stage_statuses=list(self._stage_statuses.values()),
                warnings=self._warnings,
            )
            self._save_index_record(
                ctx,
                status="completed_with_warnings" if failed_indexes else "completed",
                error=(
                    "Some indexes failed: " + ", ".join(sorted(failed_indexes.keys()))
                    if failed_indexes
                    else None
                ),
                manifest=manifest.model_dump(mode="json"),
            )
            self._emit(stage, "completed", "Manifest built")
            return manifest.model_copy(
                update={"stage_statuses": list(self._stage_statuses.values())}
            )
        except DeepSearchError:
            raise
        except Exception as exc:
            self._emit(stage, "failed", str(exc))
            raise DeepSearchError(
                DS_PIPELINE_STAGE_ERROR, str(exc), stage_or_node=stage, retryable=False
            ) from exc

    def _emit(
        self,
        stage: str,
        status: str,
        message: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        now = time.time()
        stage_status = self._stage_statuses[stage]
        if status == "started":
            self._stage_start_times[stage] = now
            stage_status.status = "running"
            stage_status.started_at = (
                stage_status.started_at or IndexEvent(stage=stage, status="started").ts
            )
        elif status in {"completed", "failed"}:
            stage_status.status = status
            stage_status.completed_at = IndexEvent(stage=stage, status="completed").ts
            if stage in self._stage_start_times:
                duration_ms = int((now - self._stage_start_times[stage]) * 1000)
                self._stats.stage_timings = [
                    timing
                    for timing in self._stats.stage_timings
                    if timing.stage != stage
                ]
                self._stats.stage_timings.append(
                    StageTiming(stage=stage, duration_ms=duration_ms)
                )
        stage_status.message = message
        stage_status.progress = progress
        event = IndexEvent(
            stage=stage, status=status, message=message, progress=progress
        )
        logger.info("[%s] %s", stage, message or status)
        if self._on_event:
            self._on_event(event)

    def _resolve_video(
        self,
        collection: Any,
        *,
        video_url: Optional[str],
        media_id: Optional[str],
        name: Optional[str],
    ):
        if media_id:
            logger.info("Resolving existing video media_id=%s", media_id)
            video = collection.get_video(media_id)
            self._save_index_record(
                {
                    "collection_id": collection.id,
                    "source_key": self.source_key,
                    "video_id": getattr(video, "id", media_id),
                },
                status="resolving",
            )
            return video
        logger.info("Uploading video from source url=%s name=%s", video_url, name or "")
        video = collection.upload(url=video_url, name=name)
        self._save_index_record(
            {
                "collection_id": collection.id,
                "source_key": self.source_key,
                "video_id": getattr(video, "id", None),
                "audio_id": getattr(video, "audio_id", None),
                "video_url": video_url,
            },
            status="uploaded",
        )
        return video

    def _index_transcript(self, video):
        cfg = self.idx_cfg.transcript
        logger.info(
            "Transcript stage method=%s engine=%s language_code=%s video_id=%s",
            cfg.method,
            cfg.engine,
            cfg.language_code or "auto",
            getattr(video, "id", ""),
        )
        try:
            video.index_spoken_words(language_code=cfg.language_code or None)
        except Exception as exc:
            msg = str(exc).lower()
            already_exists = (
                "already indexed" in msg
                or "already exists" in msg
                or "spoken word index" in msg
                and "exists" in msg
            )
            if not already_exists:
                raise
            logger.info(
                "Transcript already present for video_id=%s", getattr(video, "id", "")
            )

    def _extract_scenes(self, video):
        se_cfg = self.idx_cfg.scene_extraction
        logger.info(
            "Scene extraction config video_id=%s shot_threshold=%s shot_frame_count=%s time_interval=%s time_frame_count=%s",
            getattr(video, "id", ""),
            se_cfg.shot_threshold,
            se_cfg.shot_frame_count,
            se_cfg.time_interval,
            se_cfg.time_frame_count,
        )
        shot_scene_id = self._extract_scene_type(
            video,
            SceneExtractionType.shot_based,
            {
                "threshold": se_cfg.shot_threshold,
                "frame_count": se_cfg.shot_frame_count,
            },
        )
        time_scene_id = self._extract_scene_type(
            video,
            SceneExtractionType.time_based,
            {"time": se_cfg.time_interval, "frame_count": se_cfg.time_frame_count},
        )
        return shot_scene_id, time_scene_id

    def _extract_scene_type(self, video, extraction_type, config):
        import re

        try:
            logger.info(
                "Requesting scene extraction video_id=%s extraction_type=%s config=%s",
                getattr(video, "id", ""),
                extraction_type,
                config,
            )
            result = video.extract_scenes(
                extraction_type=extraction_type, extraction_config=config
            )
            return result.id if hasattr(result, "id") else result
        except Exception as exc:
            msg = str(exc)
            if "already exists" in msg:
                match = re.search(r"already exists with id ([a-zA-Z0-9_-]+)", msg)
                if match:
                    logger.info(
                        "Reusing existing scene collection extraction_type=%s scene_id=%s",
                        extraction_type,
                        match.group(1),
                    )
                    return match.group(1)
            raise

    def _build_detector_provider(self):
        provider_name = getattr(
            self.idx_cfg.object_detection,
            "provider",
            self.idx_cfg.object_detection.backend,
        )
        if self.idx_cfg.object_detection.mode != "local":
            logger.info(
                "Object detection disabled because mode=%s",
                self.idx_cfg.object_detection.mode,
            )
            return None
        if provider_name == "rtdetr_v2":
            logger.info(
                "Preparing local object detector provider=%s threshold=%.2f batch_size=%s",
                provider_name,
                self.idx_cfg.object_detection.threshold,
                self.idx_cfg.object_detection.batch_size,
            )
            return RTDetrDetector(
                threshold=self.idx_cfg.object_detection.threshold,
                batch_size=self.idx_cfg.object_detection.batch_size,
            )
        raise ValueError(f"Unsupported detector provider '{provider_name}'")

    def _run_object_detection(
        self,
        collection_id,
        video,
        time_scene_id,
        work_dir,
        provider,
    ):
        if provider is None:
            logger.info(
                "Skipping local object detection for video_id=%s",
                getattr(video, "id", ""),
            )
            return []
        frame_scenes = video.get_scene_collection(time_scene_id).scenes
        logger.info(
            "Running local object detection frame_scene_collection=%s windows=%s",
            time_scene_id,
            len(frame_scenes),
        )
        frames = []
        for fs in frame_scenes:
            if fs.frames:
                for frame in fs.frames:
                    frames.append(
                        {
                            "video_id": getattr(video, "id", ""),
                            "frame_url": frame.url,
                            "frame_time": getattr(
                                frame, "frame_time", getattr(frame, "time", fs.start)
                            ),
                            "start": float(fs.start),
                            "end": float(fs.end),
                        }
                    )

        existing = []
        if self.index_artifact_store:
            saved = self.index_artifact_store.load_index_artifact(
                collection_id, getattr(video, "id", ""), "vision_metadata"
            )
            if isinstance(saved, list):
                existing = saved

        existing_by_key = {
            self._frame_key(item.get("frame_url"), item.get("time")): item
            for item in existing
            if isinstance(item, dict)
        }
        pending = [
            f
            for f in frames
            if self._frame_key(f.get("frame_url"), f.get("frame_time"))
            not in existing_by_key
        ]
        logger.info(
            "Object detection resume status existing=%s pending=%s total=%s",
            len(existing_by_key),
            len(pending),
            len(frames),
        )

        chunk_size = max(1, int(self.idx_cfg.object_detection.resume_chunk_size))
        with tqdm(
            total=len(pending), desc="Object detection", unit="frame", leave=False
        ) as pbar:
            for i in range(0, len(pending), chunk_size):
                chunk = pending[i : i + chunk_size]
                detections = provider.detect_batch(
                    [
                        {"frame_url": f["frame_url"], "frame_time": f["frame_time"]}
                        for f in chunk
                    ],
                    show_progress=False,
                    progress_cb=pbar.update,
                )
                detected_by_key = {
                    self._frame_key(det.frame_url, det.frame_time): det
                    for det in detections
                }
                for frame in chunk:
                    key = self._frame_key(
                        frame.get("frame_url"), frame.get("frame_time")
                    )
                    det = detected_by_key.get(key)
                    existing_by_key[key] = {
                        "video_id": frame.get("video_id"),
                        "start": frame.get("start"),
                        "end": frame.get("end"),
                        "time": frame.get("frame_time"),
                        "frame_url": frame.get("frame_url"),
                        "provider": "rtdetr_v2",
                        "object_detection": [
                            {
                                "label": item.label,
                                "score": item.score,
                                "confidence": item.score,
                                "bbox": item.box,
                            }
                            for item in (det.detections if det else [])
                        ],
                    }
                vision_metadata = sorted(
                    existing_by_key.values(),
                    key=lambda x: (
                        float(x.get("time", 0)),
                        str(x.get("frame_url", "")),
                    ),
                )
                self._save_artifact(
                    {
                        "collection_id": collection_id,
                        "video_id": getattr(video, "id", ""),
                    },
                    "vision_metadata",
                    vision_metadata,
                )
                logger.info(
                    "Object detection checkpoint saved processed=%s/%s",
                    min(i + len(chunk), len(pending)) + len(existing),
                    len(frames),
                )

        vision_metadata = sorted(
            existing_by_key.values(),
            key=lambda x: (float(x.get("time", 0)), str(x.get("frame_url", ""))),
        )
        vision_path = os.path.join(work_dir, "vision_metadata.json")
        with open(vision_path, "w", encoding="utf-8") as handle:
            json.dump(vision_metadata, handle)
        logger.info(
            "Saved object detection output path=%s frame_windows=%s",
            vision_path,
            len(vision_metadata),
        )
        return vision_metadata

    @staticmethod
    def _frame_key(frame_url: Any, frame_time: Any) -> str:
        return f"{frame_url}|{float(frame_time or 0):.6f}"

    def _get_transcript_for_vlm(self, video):
        raw = video.get_transcript()
        return [
            {
                "start_time": float(s.get("start", 0)),
                "end_time": float(s.get("end", 0)),
                "transcript": s.get("text", ""),
            }
            for s in raw
            if s.get("text", "") != "-"
        ]

    def _run_vlm(
        self, transcript, vision_metadata, shot_scenes, frame_scenes, work_dir
    ):
        vlm_cfg = self.idx_cfg.vlm
        prompt_file = vlm_cfg.prompt_file
        if prompt_file and not os.path.isabs(prompt_file):
            pkg_dir = Path(__file__).resolve().parent.parent
            candidate = pkg_dir / prompt_file
            if candidate.exists():
                prompt_file = str(candidate)
        extractor = VLMExtractor(
            config={
                "llm": self.llm_cfg.model_dump(),
                "model": self.llm_cfg.models.indexing.scene_enrichment,
                "batch_size": vlm_cfg.batch_size,
                "llm_max_images": vlm_cfg.llm_max_images,
                "object_threshold": self.idx_cfg.object_detection.threshold,
                "temperature": vlm_cfg.temperature,
                "thinking_budget": vlm_cfg.thinking_budget,
                "retry_attempts": vlm_cfg.retry_attempts,
                "retry_backoff_sec": vlm_cfg.retry_backoff_sec,
                "max_concurrent_llm_calls": vlm_cfg.max_concurrent_llm_calls,
                "prompt_file": prompt_file,
            },
            transcript=transcript,
            vision_metadata=vision_metadata,
            output_dir=work_dir,
        )
        logger.info(
            "Running VLM enrichment model=%s prompt_file=%s shot_scenes=%s frame_scenes=%s transcript_segments=%s vision_windows=%s",
            self.llm_cfg.models.indexing.scene_enrichment,
            prompt_file or "package-default",
            len(shot_scenes),
            len(frame_scenes),
            len(transcript),
            len(vision_metadata),
        )
        return asyncio.run(extractor.process_scenes(shot_scenes, frame_scenes))

    def _load_compiled_scenes(self, path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def _maybe_restore_artifacts_for_reindex(self, ctx: Dict[str, Any]) -> None:
        if not self.index_artifact_store:
            return
        collection_id = ctx.get("collection_id")
        video_id = ctx.get("video_id")
        if not collection_id or not video_id:
            return
        compiled = self.index_artifact_store.load_index_artifact(
            collection_id, video_id, "compiled_scenes"
        )
        if not isinstance(compiled, list) or not compiled:
            return
        subplot = self.index_artifact_store.load_index_artifact(
            collection_id, video_id, "subplot_summary"
        )
        ctx["compiled_scenes"] = compiled
        ctx["subplot_data"] = subplot if isinstance(subplot, dict) else None
        ctx["_restored_for_reindex"] = True

    def _run_subplot_summary(self, compiled_scenes, work_dir):
        vlm_cfg = self.idx_cfg.vlm
        summarizer = SubplotSummarizer(
            config={
                "llm": self.llm_cfg.model_dump(),
                "model": self.llm_cfg.models.indexing.final_summary,
                "subplot_chunk_size": vlm_cfg.subplot_chunk_size,
                "temperature": vlm_cfg.temperature,
                "thinking_budget": vlm_cfg.thinking_budget,
                "batch_size": vlm_cfg.batch_size,
                "retry_attempts": vlm_cfg.retry_attempts,
                "retry_backoff_sec": vlm_cfg.retry_backoff_sec,
                "max_concurrent_llm_calls": vlm_cfg.max_concurrent_llm_calls,
                "prompt_file": vlm_cfg.prompt_file,
            },
            output_dir=work_dir,
        )
        logger.info(
            "Running subplot/final summarization model=%s compiled_scenes=%s prompt_file=%s",
            self.llm_cfg.models.indexing.final_summary,
            len(compiled_scenes),
            vlm_cfg.prompt_file or "package-default",
        )
        return asyncio.run(summarizer.summarize(compiled_scenes))

    def _create_indexes(self, video, compiled_scenes, subplot_data):
        logger.info(
            "Writing VideoDB indexes video_id=%s compiled_scenes=%s has_subplot=%s overwrite_existing=%s",
            getattr(video, "id", ""),
            len(compiled_scenes or []),
            bool(subplot_data),
            self.idx_cfg.overwrite_existing_indexes,
        )
        indexer = VideoDBIndexer(
            video,
            config={
                "overwrite_existing_indexes": self.idx_cfg.overwrite_existing_indexes,
                "retry_failed_indexes": True,
            },
        )
        return indexer.create_indexes(compiled_scenes, subplot_data)

    def _extract_and_store_metadata(self, compiled_scenes, collection_id, video_id):
        objects = set()
        shot_types = set()
        emotions = set()
        for scene in compiled_scenes:
            for obj in scene.get("detected_objects", []):
                if isinstance(obj, str):
                    objects.add(obj)
            for obj in scene.get("objects", []):
                if isinstance(obj, str):
                    objects.add(obj)
            shot_type = scene.get("shot_type")
            if isinstance(shot_type, str) and shot_type:
                shot_types.add(shot_type)
            emotion = scene.get("emotion")
            if isinstance(emotion, dict):
                value = emotion.get("emotion_type") or emotion.get("type")
                if isinstance(value, str) and value:
                    emotions.add(value)
            elif isinstance(emotion, str) and emotion:
                emotions.add(emotion)
        self.metadata_store.save_metadata(
            collection_id,
            video_id,
            {
                "objects": sorted(objects),
                "shot_types": sorted(shot_types),
                "emotions": sorted(emotions),
            },
        )

    def _attach_resume_details(self, exc: DeepSearchError, ctx: Dict[str, Any]) -> None:
        exc.details = {
            **(exc.details or {}),
            "manifest_id": ctx.get("manifest_id"),
            "collection_id": ctx.get("collection_id"),
            "video_id": ctx.get("video_id"),
            "resume_hint": (
                f"Re-run index_video(media_id='{ctx.get('video_id')}', collection_id='{ctx.get('collection_id')}')"
                if ctx.get("video_id")
                else None
            ),
            "checkpoint_path": os.path.join(
                ctx.get("work_dir", ""), "index_checkpoint.json"
            )
            if ctx.get("work_dir")
            else None,
        }

    def _write_checkpoint(
        self,
        ctx: Dict[str, Any],
        *,
        current_stage: str,
        failed: bool = False,
        error: Optional[str] = None,
    ) -> None:
        work_dir = ctx.get("work_dir")
        if not work_dir:
            return
        checkpoint = {
            "manifest_id": ctx.get("manifest_id"),
            "collection_id": ctx.get("collection_id"),
            "video_id": ctx.get("video_id"),
            "current_stage": current_stage,
            "failed": failed,
            "error": error,
            "warnings": list(self._warnings),
        }
        path = os.path.join(work_dir, "index_checkpoint.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2, ensure_ascii=False)
        self._save_artifact(ctx, "index_checkpoint", checkpoint)

    def _save_index_record(
        self,
        ctx: Dict[str, Any],
        *,
        status: str,
        error: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (
            not self.index_record_store
            or not ctx.get("collection_id")
            or not ctx.get("source_key")
        ):
            return
        existing = (
            self.index_record_store.load_index_record(
                ctx["collection_id"], ctx["source_key"]
            )
            or {}
        )
        record = {
            **existing,
            "collection_id": ctx.get("collection_id"),
            "source_key": ctx.get("source_key"),
            "source_url": ctx.get("video_url") or existing.get("source_url"),
            "video_id": ctx.get("video_id") or existing.get("video_id"),
            "audio_id": ctx.get("audio_id") or existing.get("audio_id"),
            "status": status,
            "error": error,
            "manifest": manifest or existing.get("manifest"),
            "checkpoint_path": os.path.join(
                ctx.get("work_dir", ""), "index_checkpoint.json"
            )
            if ctx.get("work_dir")
            else existing.get("checkpoint_path"),
        }
        self.index_record_store.save_index_record(
            ctx["collection_id"], ctx["source_key"], record
        )

    def _save_artifact(
        self,
        ctx: Dict[str, Any],
        artifact_name: str,
        payload: Dict[str, Any] | list[Any],
    ) -> None:
        if (
            not self.index_artifact_store
            or not ctx.get("collection_id")
            or not ctx.get("video_id")
        ):
            return
        self.index_artifact_store.save_index_artifact(
            ctx["collection_id"],
            ctx["video_id"],
            artifact_name,
            payload,
        )
