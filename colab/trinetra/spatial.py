from __future__ import annotations

from typing import Any, Protocol

from .models import Detection, Frame, SpatialResult


class DepthProvider(Protocol):
    def estimate(self, frame: Frame, detection: Detection) -> tuple[float | None, float]: ...


class SpatialEngine:
    def __init__(self, depth_provider: DepthProvider | None = None) -> None:
        self.depth_provider = depth_provider

    def enrich(self, frame: Frame, detections: list[Detection]) -> tuple[list[Detection], dict[str, Any]]:
        output: list[Detection] = []
        spatial_results: list[SpatialResult] = []
        for detection in detections:
            result = self.analyze(frame, detection)
            spatial_results.append(result)
            data = self._data(detection)
            data.update(
                distance_m=result.distance_m,
                distance_confidence=result.confidence,
                distance_source=result.source,
                direction=result.direction,
                path_overlap=result.path_overlap,
                path_relevance=result.path_relevance,
            )
            output.append(Detection(**data))
        return output, {
            "objects": [item.to_dict() for item in spatial_results],
            "depth_available": self.depth_provider is not None,
            "corridor_objects": sum(1 for item in spatial_results if item.path_relevance > 0.5),
        }

    def analyze(self, frame: Frame, detection: Detection) -> SpatialResult:
        position_x = detection.center_x / max(1.0, float(frame.width)) if frame.width else 0.5
        position_y = detection.center_y / max(1.0, float(frame.height)) if frame.height else 0.5
        direction = self._direction(position_x)
        distance, confidence, source = self._distance(frame, detection)
        corridor_left, corridor_right = self._corridor_bounds(position_y)
        object_left = max(0.0, position_x - detection.width / max(1.0, frame.width) / 2.0)
        object_right = min(1.0, position_x + detection.width / max(1.0, frame.width) / 2.0)
        overlap = self._overlap((object_left, object_right), (corridor_left, corridor_right))
        path_relevance = overlap * (1.0 if position_y > 0.2 else 0.6)
        return SpatialResult(
            distance_m=distance,
            confidence=confidence,
            direction=direction,
            position_x=position_x,
            position_y=position_y,
            source=source,
            zone="LEFT" if position_x < 0.35 else "RIGHT" if position_x > 0.65 else "CENTER",
            path_overlap=overlap,
            path_relevance=path_relevance,
            depth_available=self.depth_provider is not None and source == "depth",
        )

    def _distance(self, frame: Frame, detection: Detection) -> tuple[float | None, float, str]:
        if self.depth_provider is not None:
            try:
                distance, confidence = self.depth_provider.estimate(frame, detection)
                if distance is not None:
                    return distance, confidence, "depth"
            except Exception:
                pass
        if detection.distance_m is not None:
            return detection.distance_m, detection.distance_confidence, detection.distance_source or "injected"
        if detection.height > 1.0:
            # Class-independent relative estimate; deliberately low confidence.
            return max(0.5, min(12.0, 1.8 * frame.height / detection.height)), 0.25, "object_size"
        return None, 0.0, "unknown"

    @staticmethod
    def _direction(position_x: float) -> str:
        if position_x < 0.2:
            return "left"
        if position_x < 0.4:
            return "slight_left"
        if position_x <= 0.6:
            return "center"
        if position_x <= 0.8:
            return "slight_right"
        return "right"

    @staticmethod
    def _corridor_bounds(position_y: float) -> tuple[float, float]:
        width = 0.26 + 0.22 * max(0.0, min(1.0, position_y))
        return 0.5 - width / 2.0, 0.5 + width / 2.0

    @staticmethod
    def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
        intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
        union = max(left[1], right[1]) - min(left[0], right[0])
        return intersection / union if union else 0.0

    @staticmethod
    def _data(detection: Detection) -> dict[str, Any]:
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
