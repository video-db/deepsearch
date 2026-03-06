from __future__ import annotations

import logging
import platform
import time
import math
from typing import Any, Dict, List

import requests
from io import BytesIO
from tqdm.auto import tqdm

from deepsearch.providers.detector.base import (
    Detection,
    DetectorProvider,
    FrameDetections,
)

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA GPU detected, using cuda")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Metal GPU detected, using mps")
            return "mps"
    except ImportError:
        pass
    logger.info("No GPU detected, using cpu")
    return "cpu"


class RTDetrDetector(DetectorProvider):
    """Object detection using RT-DETR v2 with automatic device selection."""

    def __init__(
        self, threshold: float = 0.85, batch_size: int = 64, device: str | None = None
    ):
        try:
            import torch
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        except ImportError as exc:
            raise ImportError(
                "Local RT-DETR detection requires optional dependencies: torch, transformers, and Pillow. "
                "Install them from requirements comments before using indexing.object_detection.mode=local."
            ) from exc

        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device or _auto_device()

        logger.info(
            "Loading RT-DETR v2 provider=%s device=%s threshold=%.2f batch_size=%s",
            "PekingU/rtdetr_v2_r50vd",
            self.device,
            self.threshold,
            self.batch_size,
        )
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            "PekingU/rtdetr_v2_r50vd"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("RT-DETR v2 loaded and ready for local object detection")

    def detect_batch(self, frames: List[Dict[str, Any]]) -> List[FrameDetections]:
        if not frames:
            return []

        from PIL import Image

        start = time.time()
        images, times, urls = [], [], []
        for f in tqdm(
            frames,
            desc="Object detection: fetching frames",
            unit="frame",
            leave=False,
        ):
            try:
                resp = requests.get(f["frame_url"], timeout=30)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                images.append(img)
                times.append(f["frame_time"])
                urls.append(f.get("frame_url"))
            except Exception as e:
                logger.warning(f"Failed to fetch frame at t={f.get('frame_time')}: {e}")

        results: List[FrameDetections] = []
        total_batches = math.ceil(len(images) / self.batch_size) if images else 0
        for i in tqdm(
            range(0, len(images), self.batch_size),
            desc="Object detection: inferencing",
            unit="batch",
            total=total_batches,
            leave=False,
        ):
            batch_imgs = images[i : i + self.batch_size]
            batch_times = times[i : i + self.batch_size]
            batch_urls = urls[i : i + self.batch_size]
            batch_dets = self._run_batch(batch_imgs)
            for t, u, dets in zip(batch_times, batch_urls, batch_dets):
                results.append(
                    FrameDetections(
                        frame_time=t,
                        frame_url=u,
                        detections=dets,
                        provider="rtdetr_v2",
                    )
                )

        elapsed = time.time() - start
        logger.info(
            "Local object detection complete frames=%s batches=%s elapsed=%.1fs",
            len(images),
            (len(images) + self.batch_size - 1) // self.batch_size,
            elapsed,
        )
        return results

    def _run_batch(self, images) -> List[List[Detection]]:
        import torch

        inputs = self.processor(images=images, do_pad=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        sizes = torch.tensor(
            [(img.height, img.width) for img in images], device=self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
        post = self.processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=self.threshold
        )

        del inputs, outputs, sizes
        if self.device == "cuda":
            torch.cuda.empty_cache()

        all_dets: List[List[Detection]] = []
        for result in post:
            frame_dets = []
            for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                frame_dets.append(
                    Detection(
                        label=self.model.config.id2label[label_id.item()],
                        score=round(score.item(), 4),
                        box=[round(c, 2) for c in box.tolist()],
                    )
                )
            all_dets.append(frame_dets)
        return all_dets
