from __future__ import annotations

import asyncio
import aiohttp
import base64
import bisect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from deepsearch.providers.llm.client_factory import (
    build_async_openai_client,
    supports_reasoning_param,
)

logger = logging.getLogger(__name__)


class VLMExtractor:
    def __init__(
        self,
        config: Dict[str, Any],
        transcript: List[Dict],
        vision_metadata: List[Dict],
        output_dir: str,
    ):
        self.config = config
        self.transcript = transcript
        self.vision_metadata = vision_metadata
        self.output_dir = output_dir

        self.model = config.get("model", "gpt-4o")
        self.batch_size = config.get("batch_size", 5)
        self.max_images = config.get("llm_max_images", 10)
        self.object_threshold = config.get("object_threshold", 0.85)
        self.temperature = config.get("temperature", 1.0)
        self.thinking_budget = config.get("thinking_budget")
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_backoff = config.get("retry_backoff_sec", 1)
        self.max_concurrent_llm_calls = max(
            1, int(config.get("max_concurrent_llm_calls", 8))
        )

        self.client = build_async_openai_client(config.get("llm", {}))
        self._supports_reasoning_param = supports_reasoning_param(self.client)
        self.prompt = self._load_prompt(config.get("prompt_file"))
        self._llm_semaphore = asyncio.Semaphore(self.max_concurrent_llm_calls)
        logger.info(
            "Initialized VLM extractor model=%s batch_size=%s max_images=%s object_threshold=%.2f retry_attempts=%s max_concurrent_llm_calls=%s",
            self.model,
            self.batch_size,
            self.max_images,
            self.object_threshold,
            self.retry_attempts,
            self.max_concurrent_llm_calls,
        )
        logger.info(
            "VLM extractor reasoning param support=%s",
            self._supports_reasoning_param,
        )

        os.makedirs(output_dir, exist_ok=True)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0

        self._index_transcript()
        self._index_vision_metadata()

    def _load_prompt(self, prompt_file: Optional[str]) -> str:
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file) as f:
                prompts = yaml.safe_load(f)
            return prompts.get("vlm_extract_prompt", self._default_prompt())

        pkg_prompt = Path(__file__).resolve().parent / "prompts" / "vlm_extract.yaml"
        if pkg_prompt.exists():
            with open(pkg_prompt) as f:
                prompts = yaml.safe_load(f)
            return prompts.get("vlm_extract_prompt", self._default_prompt())

        return self._default_prompt()

    def _default_prompt(self) -> str:
        return (
            "Analyze this video scene and provide a JSON response with: "
            "scene_description, action, location, objects, "
            "emotion, topic, shot_type, object_description, song_detection."
        )

    def _index_transcript(self):
        self._sorted_transcript = sorted(
            self.transcript, key=lambda x: x.get("start_time", 0)
        )
        self._transcript_starts = [
            t.get("start_time", 0) for t in self._sorted_transcript
        ]
        self._transcript_ends = [t.get("end_time", 0) for t in self._sorted_transcript]

    def _index_vision_metadata(self):
        self._sorted_vision = sorted(
            self.vision_metadata, key=lambda x: x.get("time", 0)
        )
        self._vision_times = [v.get("time", 0) for v in self._sorted_vision]

    async def process_scenes(
        self, shot_scenes: List, frame_scenes: List
    ) -> Dict[str, Any]:
        start_time = time.time()
        total_scenes = len(shot_scenes)
        batch_size = self.batch_size
        total_batches = (total_scenes + batch_size - 1) // batch_size

        logger.info(
            f"Processing {total_scenes} scenes with VLM in {total_batches} batches"
        )

        compiled = []
        async with aiohttp.ClientSession() as session:
            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_scenes)
                batch_scenes = shot_scenes[batch_start:batch_end]

                tasks = [
                    self._process_scene(scene, frame_scenes, session)
                    for scene in batch_scenes
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Scene {batch_start + i} failed: {result}")
                    elif result:
                        compiled.append(result)

        compiled.sort(key=lambda s: s.get("start", 0))
        processing_time = time.time() - start_time

        compiled_path = os.path.join(self.output_dir, "compiled_scenes.json")
        with open(compiled_path, "w", encoding="utf-8") as f:
            json.dump(compiled, f, indent=2, ensure_ascii=False)

        logger.info(
            f"VLM extraction: {len(compiled)}/{total_scenes} scenes in {processing_time:.1f}s"
        )

        return {
            "compiled_path": compiled_path,
            "scenes_processed": len(compiled),
            "scenes_failed": total_scenes - len(compiled),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": self.total_reasoning_tokens,
            "processing_time": processing_time,
        }

    async def _process_scene(self, scene, frame_scenes, session):
        start, end = scene.start, scene.end
        frame_urls = self._get_frame_urls(scene, frame_scenes)
        if not frame_urls:
            logger.warning(
                "Skipping scene %.2f-%.2f because no frames were found", start, end
            )
            return None

        images = await self._fetch_images(session, frame_urls[: self.max_images])
        if not images:
            logger.warning(
                "Skipping scene %.2f-%.2f because no frame images could be fetched",
                start,
                end,
            )
            return None

        transcript_text = self._get_transcript(start, end)
        objects = self._get_objects(start, end)

        result, pt, ct, rt = await self._call_vlm(images, transcript_text, objects)
        logger.info(
            "VLM scene processed start=%.2f end=%.2f images=%s transcript_chars=%s detected_objects=%s",
            start,
            end,
            len(images),
            len(transcript_text),
            len(objects),
        )
        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        self.total_reasoning_tokens += rt

        result["start"] = start
        result["end"] = end
        result["transcript"] = transcript_text
        result["detected_objects"] = objects
        return result

    def _get_frame_urls(self, scene, frame_scenes):
        start, end = scene.start, scene.end
        urls = []
        for fs in frame_scenes:
            if fs.start < end and fs.end > start and fs.frames:
                urls.append(fs.frames[0].url)
        return urls

    def _get_transcript(self, start, end):
        left = bisect.bisect_left(self._transcript_ends, start)
        texts = []
        for i in range(left, len(self._sorted_transcript)):
            entry = self._sorted_transcript[i]
            if entry.get("start_time", 0) >= end:
                break
            texts.append(entry.get("transcript", ""))
        return " ".join(texts)

    def _get_objects(self, start, end):
        left = bisect.bisect_left(self._vision_times, start)
        right = bisect.bisect_right(self._vision_times, end)
        objects = set()
        for i in range(left, right):
            for det in self._sorted_vision[i].get("object_detection", []):
                if det.get("score", 0) >= self.object_threshold:
                    objects.add(det.get("label", ""))
        return list(objects)

    async def _fetch_images(self, session, urls):
        async def fetch_one(url):
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
                    return base64.b64encode(data).decode()
            except Exception:
                return None

        results = await asyncio.gather(*[fetch_one(u) for u in urls])
        return [r for r in results if r]

    async def _call_vlm(self, images, transcript, objects):
        content = f"\nTranscript: {transcript}\nDetected objects: {objects}"
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.prompt + content}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                    for img in images
                ],
            }
        ]

        for attempt in range(self.retry_attempts):
            try:
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                }
                if self.temperature:
                    api_params["temperature"] = self.temperature
                if self.thinking_budget and self._supports_reasoning_param:
                    api_params["reasoning"] = {"effort": self.thinking_budget}

                async with self._llm_semaphore:
                    resp = await self.client.chat.completions.create(**api_params)
                if not resp or not resp.choices:
                    raise ValueError("Empty response")

                data = json.loads(resp.choices[0].message.content)
                pt = getattr(resp.usage, "prompt_tokens", 0) or 0
                ct = getattr(resp.usage, "completion_tokens", 0) or 0
                rt = 0
                if (
                    hasattr(resp.usage, "output_tokens_details")
                    and resp.usage.output_tokens_details
                ):
                    rt = (
                        getattr(resp.usage.output_tokens_details, "reasoning_tokens", 0)
                        or 0
                    )
                return data, pt, ct, rt

            except Exception as e:
                logger.warning(f"VLM attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_backoff * (attempt + 1))

        raise RuntimeError(f"VLM failed after {self.retry_attempts} attempts")
