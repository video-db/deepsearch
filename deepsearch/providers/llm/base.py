from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class Usage:
    __slots__ = ("input_tokens", "output_tokens", "reasoning_tokens")

    def __init__(
        self, input_tokens: int = 0, output_tokens: int = 0, reasoning_tokens: int = 0
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse:
    __slots__ = ("data", "usage", "raw_model_id")

    def __init__(
        self,
        data: Dict[str, Any],
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        raw_model_id: str = "",
    ):
        self.data = data
        self.usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        self.raw_model_id = raw_model_id

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens

    @property
    def reasoning_tokens(self) -> int:
        return self.usage.reasoning_tokens


class LLMProvider(ABC):
    @abstractmethod
    def generate_json(
        self, *, task: str, prompt: str, model: str, options: Dict[str, Any]
    ) -> LLMResponse: ...

    @abstractmethod
    async def generate_json_async(
        self, *, task: str, prompt: str, model: str, options: Dict[str, Any]
    ) -> LLMResponse: ...

    def generate(self, prompt: str, *, model: Optional[str] = None) -> LLMResponse:
        return self.generate_json(
            task="generic", prompt=prompt, model=model or "", options={}
        )

    async def generate_async(
        self, prompt: str, *, model: Optional[str] = None
    ) -> LLMResponse:
        return await self.generate_json_async(
            task="generic", prompt=prompt, model=model or "", options={}
        )


class NodeLLM:
    """Thin wrapper consumed by retrieval nodes.

    Adapts the LLMProvider interface to the (data, inp, out, total) tuple that
    every retrieval node expects.
    """

    def __init__(self, provider: LLMProvider, model: str, *, task: str):
        self.provider = provider
        self.model = model
        self.task = task

    def generate(self, prompt: str) -> Tuple[Optional[Dict], int, int, int]:
        try:
            resp = self.provider.generate_json(
                task=self.task, prompt=prompt, model=self.model, options={}
            )
            return resp.data, resp.input_tokens, resp.output_tokens, resp.total_tokens
        except Exception:
            return None, 0, 0, 0

    async def generate_async(self, prompt: str) -> Tuple[Optional[Dict], int, int, int]:
        try:
            resp = await self.provider.generate_json_async(
                task=self.task, prompt=prompt, model=self.model, options={}
            )
            return resp.data, resp.input_tokens, resp.output_tokens, resp.total_tokens
        except Exception:
            return None, 0, 0, 0
