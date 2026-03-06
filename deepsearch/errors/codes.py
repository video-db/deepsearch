from __future__ import annotations

from typing import Any, Dict, Optional


DS_AUTH_ERROR = "DS_AUTH_ERROR"
DS_PROVIDER_ERROR = "DS_PROVIDER_ERROR"
DS_TIMEOUT_ERROR = "DS_TIMEOUT_ERROR"
DS_MISSING_INDEX_ERROR = "DS_MISSING_INDEX_ERROR"
DS_INVALID_PLAN_ERROR = "DS_INVALID_PLAN_ERROR"
DS_VALIDATION_ERROR = "DS_VALIDATION_ERROR"
DS_PIPELINE_STAGE_ERROR = "DS_PIPELINE_STAGE_ERROR"


class DeepSearchError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        stage_or_node: str = "",
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.stage_or_node = stage_or_node
        self.retryable = retryable
        self.details = details or {}
        super().__init__(f"[{code}] {message}")
