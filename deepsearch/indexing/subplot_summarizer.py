from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
import yaml

from deepsearch.providers.llm.client_factory import (
    build_async_openai_client,
    supports_reasoning_param,
)

logger = logging.getLogger(__name__)


class Subplot(BaseModel):
    start: float
    end: float
    summary: str


class PlotSummary(BaseModel):
    subplots: List[Subplot]
    final_summary: str


class SubplotSummarizer:
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = output_dir

        self.model = config.get("model", "gpt-4o")
        self.chunk_size = config.get("subplot_chunk_size", 250)
        self.temperature = config.get("temperature", 1.0)
        self.thinking_budget = config.get("thinking_budget")
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_backoff = config.get("retry_backoff_sec", 1)
        self.batch_size = config.get("batch_size", 5)
        self.max_concurrent_llm_calls = max(
            1, int(config.get("max_concurrent_llm_calls", 8))
        )

        self.client = build_async_openai_client(config.get("llm", {}))
        self._supports_reasoning_param = supports_reasoning_param(self.client)
        self.prompt = self._load_prompt(config.get("prompt_file"))
        self.combine_prompt = self._load_combine_prompt(config.get("prompt_file"))
        self._llm_semaphore = asyncio.Semaphore(self.max_concurrent_llm_calls)
        logger.info(
            "Initialized subplot summarizer model=%s chunk_size=%s batch_size=%s retry_attempts=%s max_concurrent_llm_calls=%s",
            self.model,
            self.chunk_size,
            self.batch_size,
            self.retry_attempts,
            self.max_concurrent_llm_calls,
        )
        logger.info(
            "Subplot summarizer reasoning param support=%s",
            self._supports_reasoning_param,
        )

        os.makedirs(output_dir, exist_ok=True)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0

    def _load_prompt(self, prompt_file: Optional[str]) -> str:
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file) as f:
                return yaml.safe_load(f).get(
                    "subplot_summary_prompt", self._default_prompt()
                )

        pkg = Path(__file__).resolve().parent / "prompts" / "vlm_extract.yaml"
        if pkg.exists():
            with open(pkg) as f:
                return yaml.safe_load(f).get(
                    "subplot_summary_prompt", self._default_prompt()
                )
        return self._default_prompt()

    def _load_combine_prompt(self, prompt_file: Optional[str]) -> str:
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file) as f:
                return yaml.safe_load(f).get(
                    "subplot_combine_prompt", self._default_combine()
                )

        pkg = Path(__file__).resolve().parent / "prompts" / "vlm_extract.yaml"
        if pkg.exists():
            with open(pkg) as f:
                return yaml.safe_load(f).get(
                    "subplot_combine_prompt", self._default_combine()
                )
        return self._default_combine()

    def _default_prompt(self):
        return (
            "Analyze the following video scenes and create subplot summaries. "
            'Output JSON: {"subplots": [{"start": 0.0, "end": 120.5, "summary": "..."}], "final_summary": "..."}'
        )

    def _default_combine(self):
        return "Merge the following subplot summaries into one coherent JSON result."

    async def summarize(self, compiled_scenes: List[Dict]) -> Dict[str, Any]:
        start_time = time.time()
        if not compiled_scenes:
            return {
                "subplot_path": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "processing_time": 0,
            }

        chunks = [
            compiled_scenes[i : i + self.chunk_size]
            for i in range(0, len(compiled_scenes), self.chunk_size)
        ]
        total_chunks = len(chunks)

        logger.info(
            f"Summarizing {len(compiled_scenes)} scenes in {total_chunks} chunks"
        )

        summaries = []
        total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
        for batch_idx in range(total_batches):
            bs = batch_idx * self.batch_size
            be = min(bs + self.batch_size, total_chunks)
            tasks = [
                self._summarize_chunk(i + bs, chunk)
                for i, chunk in enumerate(chunks[bs:be])
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Chunk failed: {r}")
                elif r:
                    summaries.append(r)

        if not summaries:
            return {
                "subplot_path": None,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "reasoning_tokens": self.total_reasoning_tokens,
                "processing_time": time.time() - start_time,
            }

        final = (
            summaries[0]
            if len(summaries) == 1
            else await self._combine_summaries(summaries)
        )
        processing_time = time.time() - start_time

        subplot_path = os.path.join(self.output_dir, "subplot_summary.json")
        with open(subplot_path, "w", encoding="utf-8") as f:
            json.dump(final.model_dump(), f, indent=2, ensure_ascii=False)

        return {
            "subplot_path": subplot_path,
            "subplot_count": len(final.subplots),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "reasoning_tokens": self.total_reasoning_tokens,
            "processing_time": processing_time,
        }

    async def _summarize_chunk(self, idx, scenes):
        content = json.dumps(scenes, ensure_ascii=False)
        for attempt in range(self.retry_attempts):
            try:
                logger.info("Summarizing chunk idx=%s scene_count=%s", idx, len(scenes))
                result, pt, ct, rt = await self._call_llm(self.prompt, content)
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                self.total_reasoning_tokens += rt
                return PlotSummary.model_validate(result)
            except Exception as e:
                logger.warning(
                    "Chunk idx=%s summarization attempt=%s failed: %s",
                    idx,
                    attempt + 1,
                    e,
                )
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_backoff * (attempt + 1))
        return None

    async def _combine_summaries(self, summaries):
        content = json.dumps([s.model_dump() for s in summaries], ensure_ascii=False)
        for attempt in range(self.retry_attempts):
            try:
                logger.info(
                    "Combining subplot summaries chunk_count=%s", len(summaries)
                )
                result, pt, ct, rt = await self._call_llm(self.combine_prompt, content)
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                self.total_reasoning_tokens += rt
                return PlotSummary.model_validate(result)
            except Exception as e:
                logger.warning(
                    "Combine summaries attempt=%s failed: %s", attempt + 1, e
                )
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_backoff * (attempt + 1))

        all_subplots = sorted(
            [sp for s in summaries for sp in s.subplots], key=lambda x: x.start
        )
        return PlotSummary(
            subplots=all_subplots,
            final_summary=summaries[-1].final_summary if summaries else "",
        )

    async def _call_llm(self, prompt, content):
        messages = [{"role": "user", "content": prompt + "\n\nScenes:\n" + content}]
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
        data = json.loads(resp.choices[0].message.content)
        rt = 0
        if (
            hasattr(resp.usage, "output_tokens_details")
            and resp.usage.output_tokens_details
        ):
            rt = getattr(resp.usage.output_tokens_details, "reasoning_tokens", 0) or 0
        return data, resp.usage.prompt_tokens, resp.usage.completion_tokens, rt
