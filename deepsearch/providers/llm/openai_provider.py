from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI, OpenAI

from deepsearch.providers.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _extract_json(text: str) -> dict:
    m = _JSON_BLOCK_RE.search(text)
    raw = m.group(1).strip() if m else text.strip()
    return json.loads(raw)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url
        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._sync = OpenAI(**kwargs)
        self._async = AsyncOpenAI(**kwargs)

    def generate_json(
        self, *, task: str, prompt: str, model: str, options: dict
    ) -> LLMResponse:
        mdl = model or "gpt-4o-2024-11-20"
        resp = self._sync.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        data = _extract_json(content)
        usage = resp.usage
        reasoning = 0
        if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
        return LLMResponse(
            data=data,
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            reasoning_tokens=reasoning,
            raw_model_id=mdl,
        )

    async def generate_json_async(
        self, *, task: str, prompt: str, model: str, options: dict
    ) -> LLMResponse:
        mdl = model or "gpt-4o-2024-11-20"
        resp = await self._async.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        data = _extract_json(content)
        usage = resp.usage
        reasoning = 0
        if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
        return LLMResponse(
            data=data,
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            reasoning_tokens=reasoning,
            raw_model_id=mdl,
        )
