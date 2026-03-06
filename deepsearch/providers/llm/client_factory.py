from __future__ import annotations

import os
from inspect import signature
from typing import Any, Dict

from openai import AsyncOpenAI


def build_async_openai_client(llm_cfg: Dict[str, Any]) -> AsyncOpenAI:
    route = llm_cfg.get("route", "vercel_ai_sdk_python")
    provider_mode = llm_cfg.get("provider_mode", "direct")
    openrouter_cfg = llm_cfg.get("openrouter", {}) or {}
    if (
        provider_mode == "openrouter"
        or openrouter_cfg.get("enabled")
        or route == "openrouter"
    ):
        api_key_env = str(openrouter_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        return AsyncOpenAI(
            api_key=os.getenv(api_key_env),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    if route == "vercel_ai_sdk_python":
        return AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
            or os.getenv("VERCEL_AI_GATEWAY_API_KEY"),
            base_url=os.getenv("VERCEL_AI_GATEWAY_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
        )
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )


def supports_reasoning_param(client: AsyncOpenAI) -> bool:
    try:
        params = signature(client.chat.completions.create).parameters
        return "reasoning" in params
    except (TypeError, ValueError):
        return False
