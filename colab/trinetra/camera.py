from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .frame_buffer import LatestFrameBuffer
from .metrics import Metrics

logger = logging.getLogger(__name__)


class MjpegCamera:
    def __init__(self, url: str, buffer: LatestFrameBuffer, metrics: Metrics, *, reconnect_s: float = 1.0) -> None:
        self.url = url
        self.buffer = buffer
        self.metrics = metrics
        self.reconnect_s = reconnect_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._last_frame_s: float | None = None
        self._error: str | None = None
        self.connected = False
        self.frames_received = 0
        self._image_lock = threading.Lock()
        self._latest_image: Any | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="trinetra-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._capture is not None:
            self._capture.release()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def status(self) -> str:
        return "CONNECTED" if self.connected else "CAMERA_ERROR" if self._error else "DISCONNECTED"

    def latest_jpeg(self) -> bytes | None:
        with self._image_lock:
            image = self._latest_image
        if image is None:
            return None
        try:
            import cv2
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            return encoded.tobytes() if ok else None
        except Exception:
            return None

    def _run(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            self._error = f"opencv unavailable: {exc}"
            return
        while not self._stop.is_set():
            try:
                logger.info("connecting to MJPEG camera: %s", self.url)
                self._capture = cv2.VideoCapture(self.url)
                if not self._capture.isOpened():
                    raise RuntimeError("MJPEG stream could not be opened")
                self.connected = True
                self._error = None
                while not self._stop.is_set():
                    ok, image = self._capture.read()
                    now = time.monotonic()
                    if not ok or image is None or getattr(image, "size", 0) == 0:
                        raise RuntimeError("invalid MJPEG frame")
                    fps = 0.0 if self._last_frame_s is None else 1.0 / max(1e-6, now - self._last_frame_s)
                    self._last_frame_s = now
                    height, width = image.shape[:2]
                    orientation = "landscape" if width > height else "portrait" if height > width else "square"
                    with self._image_lock:
                        self._latest_image = image
                    self.buffer.publish(image, captured_at=now, width=width, height=height, orientation=orientation)
                    self.metrics.record_camera(fps)
                    self.frames_received += 1
            except Exception as exc:
                self._error = str(exc)
                self.connected = False
                logger.warning("camera error; reconnecting: %s", exc)
                self._stop.wait(self.reconnect_s)
            finally:
                if self._capture is not None:
                    self._capture.release()
                    self._capture = None
