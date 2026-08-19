from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import Settings
from .models import Frame, FrameQuality

logger = logging.getLogger(__name__)


class FrameProcessor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def process(self, frame: Frame) -> tuple[Frame, FrameQuality]:
        try:
            image = self._orient(frame.image, frame.orientation)
            if image is None or getattr(image, "size", 0) == 0:
                return frame, FrameQuality(valid=False, error="empty frame")
            quality = self.quality(image)
            return Frame(
                frame_id=frame.frame_id,
                image=image,
                captured_at=frame.captured_at,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                orientation=frame.orientation,
            ), quality
        except Exception as exc:
            logger.exception("frame preprocessing failed")
            return frame, FrameQuality(valid=False, error=str(exc))

    @staticmethod
    def _orient(image: Any, orientation: str) -> Any:
        # The stream's pixel matrix is already decoded. Rotation can be supplied by
        # the camera adapter later; no arbitrary rotation is applied here.
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            return None
        return image

    @staticmethod
    def quality(image: Any) -> FrameQuality:
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            brightness = float(np.mean(gray)) / 255.0
            contrast = float(np.std(gray)) / 128.0
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_score = min(1.0, blur / 300.0)
            visibility = max(0.0, min(1.0, 0.55 * min(1.0, contrast) + 0.45 * brightness))
            return FrameQuality(
                valid=True,
                brightness=brightness,
                contrast=contrast,
                blur=blur_score,
                visibility=visibility,
            )
        except Exception as exc:
            return FrameQuality(valid=False, error=str(exc))
