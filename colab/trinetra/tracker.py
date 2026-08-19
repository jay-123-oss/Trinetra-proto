from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

from .models import Detection, MotionState


@dataclass(slots=True)
class Track:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    previous_center_x: float
    previous_center_y: float
    last_seen: float
    first_seen: float
    confidence_history: list[float] = field(default_factory=list)
    distance_history: list[float] = field(default_factory=list)
    risk_history: list[float] = field(default_factory=list)
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class Tracker:
    """Dependency-light tracker for the prototype; a ByteTrack adapter can replace it."""

    def __init__(self, *, iou_threshold: float = 0.25, timeout_s: float = 1.0) -> None:
        self.iou_threshold = iou_threshold
        self.timeout_s = timeout_s
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], *, now_s: float | None = None) -> list[Detection]:
        now = time.monotonic() if now_s is None else now_s
        matches: set[int] = set()
        output: list[Detection] = []
        for detection in detections:
            best_id = None
            best_iou = self.iou_threshold
            for track_id, track in self._tracks.items():
                if track_id in matches or track.class_name != detection.class_name:
                    continue
                score = self._iou(track.bbox, detection.bbox)
                if score > best_iou:
                    best_iou, best_id = score, track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                track = Track(best_id, detection.class_name, detection.bbox, detection.center_x, detection.center_y, detection.center_x, detection.center_y, now, now)
            else:
                track = self._tracks[best_id]
                dt = max(1e-3, now - track.last_seen)
                track.previous_center_x, track.previous_center_y = track.center_x, track.center_y
                track.center_x, track.center_y = detection.center_x, detection.center_y
                track.bbox = detection.bbox
                track.velocity_x = (track.center_x - track.previous_center_x) / dt
                track.velocity_y = (track.center_y - track.previous_center_y) / dt
                track.last_seen = now
            track.confidence_history = (track.confidence_history + [detection.confidence])[-20:]
            if detection.distance_m is not None:
                track.distance_history = (track.distance_history + [detection.distance_m])[-20:]
            self._tracks[best_id] = track
            matches.add(best_id)
            output.append(replace(
                detection,
                track_id=best_id,
                velocity_x=track.velocity_x,
                velocity_y=track.velocity_y,
            ))
        self._tracks = {key: value for key, value in self._tracks.items() if now - value.last_seen <= self.timeout_s}
        return output

    def state(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    @staticmethod
    def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = left
        bx1, by1, bx2, by2 = right
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0
