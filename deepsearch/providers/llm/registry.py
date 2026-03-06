from __future__ import annotations

import os
from typing import Any, Callable, Dict

from deepsearch.config.schema import LLMConfig
from deepsearch.providers.llm.base import LLMProvider
from deepsearch.providers.llm.openai_provider import OpenAIProvider

_LLM_PROVIDERS: Dict[str, Callable[[LLMConfig], LLMProvider]] = {}


def register_llm_provider(
    name: str, factory: Callable[[LLMConfig], LLMProvider]
) -> None:
    _LLM_PROVIDERS[name] = factory


def create_llm_provider(cfg: LLMConfig) -> LLMProvider:
    route = cfg.route
    if cfg.provider_mode == "openrouter" or bool(cfg.openrouter.get("enabled")):
        route = "openrouter"
    if route not in _LLM_PROVIDERS:
        raise ValueError(f"Unknown llm.route '{route}'")
    return _LLM_PROVIDERS[route](cfg)


def _openai_factory(cfg: LLMConfig) -> LLMProvider:
    return OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )


def _vercel_factory(cfg: LLMConfig) -> LLMProvider:
    return OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VERCEL_AI_GATEWAY_API_KEY"),
        base_url=os.getenv("VERCEL_AI_GATEWAY_BASE_URL")
        or os.getenv("OPENAI_BASE_URL"),
    )


def _openrouter_factory(cfg: LLMConfig) -> LLMProvider:
    api_key_env = str(cfg.openrouter.get("api_key_env", "OPENROUTER_API_KEY"))
    return OpenAIProvider(
        api_key=os.getenv(api_key_env),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


register_llm_provider("openai", _openai_factory)
register_llm_provider("vercel_ai_sdk_python", _vercel_factory)
register_llm_provider("openrouter", _openrouter_factory)
