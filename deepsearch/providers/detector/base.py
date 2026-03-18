from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    score: float
    box: Optional[List[float]] = None


class FrameDetections(BaseModel):
    frame_time: float
    frame_url: Optional[str] = None
    detections: List[Detection]
    provider: str = "unknown"


class DetectorProvider(ABC):
    @abstractmethod
    def detect_batch(
        self,
        frames: List[Dict[str, Any]],
        *,
        show_progress: bool = True,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> List[FrameDetections]:
        """Detect objects in a batch of frames.

        Each frame dict must contain 'frame_url' and 'frame_time'.
        """
        ...
