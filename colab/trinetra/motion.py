from __future__ import annotations

from collections import defaultdict, deque

from .models import Detection, MotionState


class MotionEngine:
    def __init__(self, *, state_window: int = 3) -> None:
        self._states: dict[int, deque[MotionState]] = defaultdict(lambda: deque(maxlen=state_window))

    def analyze(self, detections: list[Detection]) -> list[Detection]:
        output: list[Detection] = []
        for detection in detections:
            state = self._classify(detection)
            if detection.track_id is not None:
                history = self._states[detection.track_id]
                history.append(state)
                if len(history) >= 2 and history[-1] != history[-2]:
                    # Require a repeated state unless the object is clearly approaching.
                    if state != MotionState.APPROACHING or history.count(state) < 2:
                        state = history[-2]
            output.append(Detection(**{**self._data(detection), "motion_state": state}))
        return output

    @staticmethod
    def _classify(detection: Detection) -> MotionState:
        if detection.velocity_mps is None and abs(detection.velocity_x) < 0.02 and abs(detection.velocity_y) < 0.02:
            return MotionState.UNKNOWN
        if detection.distance_m is not None and detection.velocity_mps is not None:
            if detection.velocity_mps > 0.05 and detection.velocity_y > 0:
                return MotionState.APPROACHING
            if detection.velocity_mps > 0.05 and detection.velocity_y < 0:
                return MotionState.MOVING_AWAY
        if abs(detection.velocity_x) > abs(detection.velocity_y):
            return MotionState.MOVING_RIGHT if detection.velocity_x > 0 else MotionState.MOVING_LEFT
        return MotionState.CROSSING if abs(detection.velocity_x) > 0.03 else MotionState.STATIONARY

    @staticmethod
    def _data(detection: Detection) -> dict:
        return {
            "track_id": detection.track_id, "class_id": detection.class_id, "class_name": detection.class_name,
            "confidence": detection.confidence, "bbox": detection.bbox, "center_x": detection.center_x,
            "center_y": detection.center_y, "width": detection.width, "height": detection.height,
            "timestamp": detection.timestamp, "distance_m": detection.distance_m,
            "distance_confidence": detection.distance_confidence, "distance_source": detection.distance_source,
            "direction": detection.direction, "path_overlap": detection.path_overlap,
            "path_relevance": detection.path_relevance, "velocity_x": detection.velocity_x,
            "velocity_y": detection.velocity_y, "velocity_mps": detection.velocity_mps,
            "motion_state": detection.motion_state, "ttc_s": detection.ttc_s,
            "collision_risk": detection.collision_risk,
        }
