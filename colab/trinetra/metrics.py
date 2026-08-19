from __future__ import annotations

import threading
from collections import deque
from typing import Any


class Metrics:
    def __init__(self, window_sizes: tuple[int, ...] = (10, 50, 100)) -> None:
        self._lock = threading.Lock()
        self._samples: deque[dict[str, float]] = deque(maxlen=max(window_sizes))
        self._window_sizes = window_sizes
        self.frames_processed = 0
        self.frames_dropped = 0
        self.vlm_calls = 0
        self.vlm_triggers = 0
        self.camera_frames = 0
        self.camera_fps = 0.0
        self.system_state = "INITIALIZING"

    def record(self, sample: dict[str, float]) -> None:
        with self._lock:
            self._samples.append(sample)
            self.frames_processed += 1

    def record_camera(self, fps: float) -> None:
        with self._lock:
            self.camera_frames += 1
            self.camera_fps = fps

    def snapshot(self, *, dropped_frames: int = 0, gpu_memory: float | None = None) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            latest = samples[-1] if samples else {}
            def average(key: str) -> float:
                values = [sample[key] for sample in samples if key in sample]
                return sum(values) / len(values) if values else 0.0
            return {
                "camera_fps": round(self.camera_fps, 2),
                "processing_fps": round(average("processing_fps"), 2),
                "yolo_latency_ms": round(average("yolo_latency_ms"), 2),
                "tracking_latency_ms": round(average("tracking_latency_ms"), 2),
                "spatial_latency_ms": round(average("spatial_latency_ms"), 2),
                "vlm_latency_ms": round(average("vlm_latency_ms"), 2),
                "safety_latency_ms": round(average("safety_latency_ms"), 2),
                "total_latency_ms": round(average("total_latency_ms"), 2),
                "network_latency_ms": round(average("network_latency_ms"), 2),
                "dropped_frames": dropped_frames,
                "processed_frames": self.frames_processed,
                "vlm_calls": self.vlm_calls,
                "vlm_trigger_rate": round(self.vlm_triggers / max(1, self.frames_processed), 4),
                "gpu_memory": gpu_memory,
                "system_state": self.system_state,
                "rolling_windows": {str(size): {key: round(sum(item.get(key, 0.0) for item in samples[-size:]) / max(1, len(samples[-size:])), 2) for key in latest} for size in self._window_sizes},
            }
