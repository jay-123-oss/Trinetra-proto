from __future__ import annotations

import threading
import time
from typing import Any

from .models import Frame


class LatestFrameBuffer:
    """At most one pending frame exists; publishing always replaces stale work."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: Frame | None = None
        self._next_id = 0
        self._dropped = 0
        self._closed = False

    def publish(self, image: Any, *, captured_at: float | None = None, width: int | None = None, height: int | None = None, orientation: str = "unknown") -> int:
        with self._condition:
            if self._closed:
                raise RuntimeError("latest-frame buffer is closed")
            self._next_id += 1
            if self._latest is not None:
                self._dropped += 1
            self._latest = Frame(
                frame_id=self._next_id,
                image=image,
                captured_at=time.monotonic() if captured_at is None else captured_at,
                width=int(width or 0),
                height=int(height or 0),
                orientation=orientation,
            )
            self._condition.notify()
            return self._next_id

    def take_latest(self, timeout_s: float = 0.25) -> Frame | None:
        with self._condition:
            if self._latest is None and not self._closed:
                self._condition.wait(timeout=timeout_s)
            frame = self._latest
            self._latest = None
            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._latest = None
            self._condition.notify_all()

    @property
    def dropped_count(self) -> int:
        with self._condition:
            return self._dropped

    @property
    def pending(self) -> bool:
        with self._condition:
            return self._latest is not None
