from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deepsearch.client import DeepSearchClient, DeepSearchSession
from deepsearch.config.schema import DeepSearchConfig
from deepsearch.errors.codes import DeepSearchError
from deepsearch.indexing.contracts import IndexEvent, IndexManifest
from deepsearch.indexing.pipeline import IndexingPipeline
from deepsearch.indexing.subplot_summarizer import SubplotSummarizer
from deepsearch.indexing.videodb_indexer import VideoDBIndexer
from deepsearch.indexing.vlm_extractor import VLMExtractor
from deepsearch.providers.llm.client_factory import supports_reasoning_param
from deepsearch.providers.llm.base import LLMProvider, LLMResponse, NodeLLM
from deepsearch.providers.detector.base import Detection, FrameDetections
from deepsearch.retrieval.contracts import (
    ClipResult,
    PageInfo,
    RetrievalResult,
    UiEvent,
)
from deepsearch.retrieval.graph import _routing_function, build_graph
from deepsearch.retrieval.helpers import registry
from deepsearch.retrieval.helpers.delta import DeltaBatch, DeltaOp, apply_batch
from deepsearch.retrieval.helpers.schema import JoinPlan, Plan, Subquery
from deepsearch.retrieval.nodes.plan_init import PlanInitLLM
from deepsearch.retrieval.state import GraphState
from deepsearch.stores.memory import (
    InMemoryIndexArtifactStore,
    InMemoryIndexRecordStore,
)
from deepsearch.stores.sqlite import SQLiteMetadataStore, SQLiteSessionStore


class MockLLMProvider(LLMProvider):
    def __init__(self, payload=None):
        self.payload = payload or {}

    def generate_json(self, *, task, prompt, model, options):
        return LLMResponse(
            data=self.payload, input_tokens=1, output_tokens=2, raw_model_id=model
        )

    async def generate_json_async(self, *, task, prompt, model, options):
        return self.generate_json(
            task=task, prompt=prompt, model=model, options=options
        )


class TestConfigContracts(unittest.TestCase):
    def test_defaults_align_with_specs(self):
        cfg = DeepSearchConfig.defaults()
        self.assertEqual(cfg.llm.route, "vercel_ai_sdk_python")
        self.assertEqual(cfg.retrieval.page_size, 10)
        self.assertEqual(cfg.indexing.object_detection.provider, "rtdetr_v2")
        self.assertEqual(cfg.retrieval.score_threshold, 0.0)
        self.assertEqual(cfg.indexing.vlm.max_concurrent_llm_calls, 8)

    def test_from_env_supports_nested_keys(self):
        with patch.dict(
            "os.environ", {"DEEPSEARCH_RETRIEVAL__PAGE_SIZE": "42"}, clear=False
        ):
            cfg = DeepSearchConfig.from_env()
        self.assertEqual(cfg.retrieval.page_size, 42)


class TestRegistryAndDelta(unittest.TestCase):
    def test_required_facets_present(self):
        self.assertEqual(
            sorted(registry.allowed_facet_names()), ["emotion", "objects", "shot_type"]
        )

    def test_plan_validation_checks_join_refs(self):
        with self.assertRaises(ValueError):
            Plan(
                subqueries=[Subquery(subquery_id="Q1", index=["location"], q="forest")],
                join_plan=JoinPlan(op="OR", subqueries=["Q2"]),
            )

    def test_apply_batch_keeps_contract(self):
        plan = Plan(
            subqueries=[Subquery(subquery_id="Q1", index=["location"], q="forest")],
            join_plan=JoinPlan(op="OR", subqueries=["Q1"]),
            metadata_filters={"shot_type": ["wide"]},
        )
        batch = DeltaBatch(
            ops=[DeltaOp(op="set_filter", facet="shot_type", values=["close-up"])]
        )
        updated = apply_batch(plan, batch)
        self.assertEqual(updated.metadata_filters["shot_type"], ["close-up"])

    def test_plan_init_normalizes_invalid_fallback_facets(self):
        llm = Mock()
        llm.generate.side_effect = [
            (
                {
                    "subqueries": [
                        {"subquery_id": "Q1", "index": ["location"], "q": "forest"}
                    ],
                    "join_plan": {"op": "OR", "subqueries": ["Q1"]},
                    "metadata_filters": {},
                    "fallback_order": ["actors", "shot_type", "unknown"],
                },
                0,
                0,
                0,
            )
        ]
        prompts = Mock()
        prompts.build_plan_init_prompt.return_value = "p"
        plan, _ = PlanInitLLM(llm, prompts).run("query")
        self.assertEqual(plan.fallback_order, ["shot_type"])


class TestGraphContracts(unittest.TestCase):
    def test_graph_routes(self):
        self.assertEqual(_routing_function(GraphState(main_query="test")), "plan_init")
        self.assertEqual(
            _routing_function(
                GraphState(main_query="test", paused_for="preview_pause")
            ),
            "interpreter",
        )
        self.assertEqual(
            _routing_function(
                GraphState(main_query="test", paused_for="clarify_pause")
            ),
            "interpreter",
        )
        self.assertIsNotNone(build_graph())


class TestProviderContracts(unittest.TestCase):
    def test_node_llm_uses_task_specific_model(self):
        provider = MockLLMProvider({"ok": True})
        llm = NodeLLM(provider, "gpt-test", task="validator")
        data, inp, out, total = llm.generate("prompt")
        self.assertEqual(data, {"ok": True})
        self.assertEqual((inp, out, total), (1, 2, 3))

    def test_reasoning_support_detected_once_from_client_signature(self):
        class WithReasoning:
            async def create(
                self, *, model=None, messages=None, response_format=None, reasoning=None
            ):
                return None

        class WithoutReasoning:
            async def create(self, *, model=None, messages=None, response_format=None):
                return None

        client_with = Mock()
        client_with.chat.completions.create = WithReasoning().create
        client_without = Mock()
        client_without.chat.completions.create = WithoutReasoning().create
        self.assertTrue(supports_reasoning_param(client_with))
        self.assertFalse(supports_reasoning_param(client_without))

    @patch("deepsearch.indexing.vlm_extractor.build_async_openai_client")
    def test_vlm_extractor_disables_reasoning_once_when_unsupported(self, mock_build):
        client = Mock()

        async def create(
            *, model=None, messages=None, response_format=None, temperature=None
        ):
            return Mock(
                choices=[Mock(message=Mock(content="{}"))],
                usage=Mock(
                    prompt_tokens=0, completion_tokens=0, output_tokens_details=None
                ),
            )

        client.chat.completions.create = create
        mock_build.return_value = client
        extractor = VLMExtractor(
            config={}, transcript=[], vision_metadata=[], output_dir=tempfile.mkdtemp()
        )
        self.assertFalse(extractor._supports_reasoning_param)

    @patch("deepsearch.indexing.subplot_summarizer.build_async_openai_client")
    def test_subplot_summarizer_disables_reasoning_once_when_unsupported(
        self, mock_build
    ):
        client = Mock()

        async def create(
            *, model=None, messages=None, response_format=None, temperature=None
        ):
            return Mock(
                choices=[
                    Mock(message=Mock(content='{"subplots": [], "final_summary": ""}'))
                ],
                usage=Mock(
                    prompt_tokens=0, completion_tokens=0, output_tokens_details=None
                ),
            )

        client.chat.completions.create = create
        mock_build.return_value = client
        summarizer = SubplotSummarizer(config={}, output_dir=tempfile.mkdtemp())
        self.assertFalse(summarizer._supports_reasoning_param)


class TestSessionPagination(unittest.TestCase):
    def _session(self, persisted):
        client = Mock()
        client.config = DeepSearchConfig.defaults()
        client._save_session = Mock()
        client._load_session = Mock(return_value=persisted)
        client._get_collection = Mock()
        return DeepSearchSession(
            session_id="s1",
            client=client,
            collection_id="c1",
            persisted_state=persisted,
        )

    def test_show_more_paginates_cached_results(self):
        persisted = {
            "session_id": "s1",
            "collection_id": "c1",
            "main_query": "query",
            "paused_for": "preview_pause",
            "page_cursor": 0,
            "page_size": 2,
            "ranked_cache": [
                ClipResult(video_id="v1", start=0, end=1, rank=1).model_dump(),
                ClipResult(video_id="v1", start=1, end=2, rank=2).model_dump(),
                ClipResult(video_id="v1", start=2, end=3, rank=3).model_dump(),
            ],
        }
        session = self._session(persisted)
        first = session.result
        self.assertEqual(len(first.clips), 2)
        nxt = session.followup(ui_event=UiEvent(type="show_more"))
        self.assertEqual(len(nxt.clips), 1)
        self.assertFalse(nxt.page.has_more)

    def test_show_more_exhaustion_returns_hint(self):
        persisted = {
            "session_id": "s1",
            "collection_id": "c1",
            "main_query": "query",
            "paused_for": "preview_pause",
            "page_cursor": 1,
            "page_size": 1,
            "ranked_cache": [
                ClipResult(video_id="v1", start=0, end=1, rank=1).model_dump()
            ],
        }
        session = self._session(persisted)
        result = session.followup(ui_event=UiEvent(type="show_more"))
        self.assertEqual(result.clips, [])
        self.assertEqual(result.debug, {"status_hint": "no_more_results"})


class TestIndexerContracts(unittest.TestCase):
    def test_overwrite_existing_indexes_uses_videodb_api(self):
        video = Mock()
        video.id = "m-1"
        video.list_scene_index.return_value = [
            {"name": "location", "scene_index_id": "idx-old"}
        ]
        video.index_scenes.return_value = "idx-new"
        indexer = VideoDBIndexer(video, config={"overwrite_existing_indexes": True})
        result = indexer.create_indexes(
            [
                {
                    "start": 0,
                    "end": 1,
                    "location": "forest",
                    "scene_description": "trees",
                    "transcript": "hello",
                    "topic": "nature",
                    "object_description": "tree",
                }
            ],
            {
                "subplots": [{"start": 0, "end": 1, "summary": "subplot"}],
                "final_summary": "final",
            },
        )
        video.delete_scene_index.assert_called_with("idx-old")
        self.assertIn("location", result["indexes"])
        self.assertEqual(result["replaced_indexes"][0]["old_index_id"], "idx-old")

    def test_failed_index_is_retried_once(self):
        video = Mock()
        video.id = "m-1"
        video.list_scene_index.return_value = []
        calls = {"action": 0}

        def index_scenes(*, scenes, name):
            if name == "action":
                calls["action"] += 1
                if calls["action"] == 1:
                    raise RuntimeError("Endpoint request timed out")
            return f"idx-{name}-{calls['action']}"

        video.index_scenes.side_effect = index_scenes
        indexer = VideoDBIndexer(
            video,
            config={"overwrite_existing_indexes": True, "retry_failed_indexes": True},
        )
        result = indexer.create_indexes(
            [
                {
                    "start": 0,
                    "end": 1,
                    "action": "run",
                    "location": "forest",
                    "scene_description": "trees",
                    "object_description": "tree",
                    "transcript": "hello",
                    "topic": "nature",
                }
            ],
            {
                "subplots": [{"start": 0, "end": 1, "summary": "subplot"}],
                "final_summary": "final",
            },
        )
        self.assertGreaterEqual(calls["action"], 2)
        self.assertIn("action", result["indexes"])
        self.assertNotIn("action", result["failed_indexes"])

    def test_detect_stage_fails_when_local_detector_deps_missing(self):
        pipeline = IndexingPipeline(config=DeepSearchConfig.defaults())
        pipeline._build_detector_provider = Mock(
            side_effect=ImportError("No module named 'torch'")
        )
        pipeline._run_object_detection = Mock(return_value=[])
        video = Mock()
        with self.assertRaises(DeepSearchError):
            pipeline._run_detect_stage(
                {
                    "video": video,
                    "time_scene_id": "sc-1",
                    "work_dir": tempfile.mkdtemp(),
                }
            )

    def test_pipeline_error_contains_resume_details_after_extract(self):
        pipeline = IndexingPipeline(config=DeepSearchConfig.defaults())
        collection = Mock()
        video = Mock()
        video.id = "m-123"
        pipeline._resolve_video = Mock(return_value=video)
        pipeline._extract_scenes = Mock(return_value=("sc-shot", "sc-time"))
        pipeline._run_transcript_stage = Mock(side_effect=RuntimeError("boom"))
        with self.assertRaises(DeepSearchError) as ctx:
            pipeline.run(collection=collection, collection_id="c-1", media_id="m-123")
        err = ctx.exception
        self.assertEqual(err.details.get("video_id"), "m-123")
        self.assertIn(
            "index_video(media_id='m-123'", err.details.get("resume_hint", "")
        )

    def test_index_artifacts_are_persisted(self):
        artifact_store = InMemoryIndexArtifactStore()
        pipeline = IndexingPipeline(
            config=DeepSearchConfig.defaults(),
            index_artifact_store=artifact_store,
        )
        pipeline._save_artifact(
            {"collection_id": "c1", "video_id": "m1"},
            "compiled_scenes",
            [{"start": 0, "end": 1}],
        )
        self.assertEqual(
            artifact_store.load_index_artifact("c1", "m1", "compiled_scenes"),
            [{"start": 0, "end": 1}],
        )

    def test_pipeline_restores_artifacts_for_reindex(self):
        artifact_store = InMemoryIndexArtifactStore()
        artifact_store.save_index_artifact(
            "c1", "m1", "compiled_scenes", [{"start": 0, "end": 1}]
        )
        artifact_store.save_index_artifact(
            "c1", "m1", "subplot_summary", {"subplots": [], "final_summary": "x"}
        )
        pipeline = IndexingPipeline(
            config=DeepSearchConfig.defaults(),
            index_artifact_store=artifact_store,
        )
        ctx = {"collection_id": "c1", "video_id": "m1"}
        pipeline._maybe_restore_artifacts_for_reindex(ctx)
        self.assertTrue(ctx.get("_restored_for_reindex"))
        self.assertEqual(ctx["compiled_scenes"], [{"start": 0, "end": 1}])

    def test_object_detection_resumes_from_saved_artifact(self):
        artifact_store = InMemoryIndexArtifactStore()
        artifact_store.save_index_artifact(
            "c1",
            "m1",
            "vision_metadata",
            [
                {
                    "video_id": "m1",
                    "start": 0.0,
                    "end": 1.0,
                    "time": 0.0,
                    "frame_url": "u0",
                    "provider": "rtdetr_v2",
                    "object_detection": [],
                }
            ],
        )
        pipeline = IndexingPipeline(
            config=DeepSearchConfig.defaults(),
            index_artifact_store=artifact_store,
        )
        frame0 = Mock(url="u0", frame_time=0.0)
        frame1 = Mock(url="u1", frame_time=1.0)
        scene = Mock(start=0.0, end=1.0, frames=[frame0, frame1])
        video = Mock(id="m1")
        video.get_scene_collection.return_value = Mock(scenes=[scene])

        provider = Mock()
        provider.detect_batch.return_value = [
            FrameDetections(
                frame_time=1.0,
                frame_url="u1",
                detections=[Detection(label="person", score=0.9, box=[0, 0, 1, 1])],
                provider="rtdetr_v2",
            )
        ]

        out = pipeline._run_object_detection(
            "c1", video, "sc-time", tempfile.mkdtemp(), provider
        )
        provider.detect_batch.assert_called_once()
        sent = provider.detect_batch.call_args.args[0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["frame_url"], "u1")
        self.assertEqual(len(out), 2)
        restored = artifact_store.load_index_artifact("c1", "m1", "vision_metadata")
        self.assertEqual(len(restored), 2)

    def test_transcript_stage_tolerates_existing_spoken_word_index(self):
        pipeline = IndexingPipeline(config=DeepSearchConfig.defaults())
        video = Mock(id="m1")
        video.index_spoken_words.side_effect = RuntimeError(
            "Invalid request: Spoken word index for video already exists."
        )
        pipeline._index_transcript(video)
        video.index_spoken_words.assert_called_once()


class TestClientContracts(unittest.TestCase):
    @patch("deepsearch.client.videodb.connect")
    def test_client_uses_sqlite_stores_by_default(self, mock_connect):
        mock_conn = Mock()
        mock_conn.get_collection.return_value = Mock(id="c1")
        mock_connect.return_value = mock_conn
        with patch.dict(
            "os.environ", {"VIDEO_DB_API_KEY": "k", "OPENAI_API_KEY": "k"}, clear=False
        ):
            client = DeepSearchClient()
        self.assertIsInstance(client._session_store, SQLiteSessionStore)
        self.assertIsInstance(client._metadata_store, SQLiteMetadataStore)

    @patch("deepsearch.client.videodb.connect")
    def test_client_passes_videodb_base_url(self, mock_connect):
        mock_conn = Mock()
        mock_conn.get_collection.return_value = Mock(id="c1")
        mock_connect.return_value = mock_conn
        with patch.dict(
            "os.environ", {"VIDEO_DB_API_KEY": "k", "OPENAI_API_KEY": "k"}, clear=False
        ):
            DeepSearchClient(base_url="https://custom.videodb.local")
        self.assertEqual(
            mock_connect.call_args.kwargs["base_url"], "https://custom.videodb.local"
        )

    @patch("deepsearch.client.videodb.connect")
    def test_completed_index_returns_stored_manifest_without_rerun(self, mock_connect):
        mock_conn = Mock()
        mock_collection = Mock(id="c1")
        mock_conn.get_collection.return_value = mock_collection
        mock_connect.return_value = mock_conn
        store = InMemoryIndexRecordStore()
        store.save_index_record(
            "c1",
            "video_url:https://example.com/video.mp4",
            {
                "status": "completed",
                "manifest": {
                    "manifest_id": "idx-1",
                    "collection_id": "c1",
                    "video_id": "m-1",
                    "indexes": {},
                    "stats": {
                        "total_scenes": 0,
                        "stage_timings": [],
                        "token_usage": {},
                        "replaced_indexes": [],
                    },
                    "stage_statuses": [],
                    "warnings": [],
                },
            },
        )
        with patch.dict(
            "os.environ", {"VIDEO_DB_API_KEY": "k", "OPENAI_API_KEY": "k"}, clear=False
        ):
            client = DeepSearchClient(index_record_store=store)
        with patch("deepsearch.client.IndexingPipeline") as pipeline_cls:
            manifest = client.index_video(
                collection_id="c1", video_url="https://example.com/video.mp4"
            )
        self.assertEqual(manifest.video_id, "m-1")
        pipeline_cls.assert_not_called()

    @patch("deepsearch.client.videodb.connect")
    def test_partial_index_resumes_with_media_id(self, mock_connect):
        mock_conn = Mock()
        mock_collection = Mock(id="c1")
        mock_conn.get_collection.return_value = mock_collection
        mock_connect.return_value = mock_conn
        store = InMemoryIndexRecordStore()
        store.save_index_record(
            "c1",
            "video_url:https://example.com/video.mp4",
            {"status": "uploaded", "video_id": "m-existing"},
        )
        with patch.dict(
            "os.environ", {"VIDEO_DB_API_KEY": "k", "OPENAI_API_KEY": "k"}, clear=False
        ):
            client = DeepSearchClient(index_record_store=store)
        fake_manifest = IndexManifest.model_validate(
            {
                "manifest_id": "idx-2",
                "collection_id": "c1",
                "video_id": "m-existing",
                "indexes": {},
                "stats": {
                    "total_scenes": 0,
                    "stage_timings": [],
                    "token_usage": {},
                    "replaced_indexes": [],
                },
                "stage_statuses": [],
                "warnings": [],
            }
        )
        with patch("deepsearch.client.IndexingPipeline") as pipeline_cls:
            pipeline = pipeline_cls.return_value
            pipeline.run.return_value = fake_manifest
            client.index_video(
                collection_id="c1", video_url="https://example.com/video.mp4"
            )
        kwargs = pipeline.run.call_args.kwargs
        self.assertIsNone(kwargs["video_url"])
        self.assertEqual(kwargs["media_id"], "m-existing")


class TestExamples(unittest.TestCase):
    def test_example_files_exist(self):
        root = Path(__file__).resolve().parent.parent
        self.assertTrue((root / "index_video.py").exists())
        self.assertTrue((root / "run_deepsearch.py").exists())
        self.assertTrue((root / "deepsearch_config.yaml").exists())
        self.assertTrue((root / ".env.sample").exists())


class TestContracts(unittest.TestCase):
    def test_retrieval_result_contract(self):
        result = RetrievalResult(
            session_id="s1",
            clips=[ClipResult(video_id="v1", start=0, end=5, rank=1)],
            waiting_for="user_input",
            page=PageInfo(page_size=10, cursor=0, next_cursor=None, has_more=False),
        )
        self.assertEqual(result.session_id, "s1")
        self.assertEqual(result.clips[0].video_id, "v1")

    def test_index_event_contract(self):
        event = IndexEvent(stage="extract", status="started")
        self.assertEqual(event.stage, "extract")
        self.assertEqual(event.status, "started")


if __name__ == "__main__":
    unittest.main()
