from __future__ import annotations

from dataclasses import replace

from .models import Detection, Hazard, MotionState


class RiskEngine:
    HIGH_IMPACT = {"vehicle", "motorcycle", "bicycle", "wall", "barrier", "stairs", "down_stairs", "road_edge", "open_drain", "pothole"}

    def enrich_detections(self, detections: list[Detection]) -> list[Detection]:
        output = []
        for detection in detections:
            ttc = self.ttc(detection.distance_m, detection.velocity_mps)
            output.append(replace(detection, ttc_s=ttc, collision_risk=self.score(detection, ttc)))
        return output

    def score(self, detection: Detection, ttc_s: float | None = None) -> float:
        path = max(0.0, min(1.0, detection.path_relevance))
        proximity = 0.0 if detection.distance_m is None else max(0.0, min(1.0, 1.0 - detection.distance_m / 4.0))
        motion = 0.25 if detection.motion_state == MotionState.APPROACHING else 0.0
        impact = 0.15 if detection.class_name.lower() in self.HIGH_IMPACT else 0.0
        ttc_component = 0.35 if ttc_s is not None and ttc_s < 2.0 else 0.0
        return round(max(0.0, min(1.0, detection.confidence * (0.40 * path + 0.35 * proximity + motion + impact + ttc_component))), 3)

    @staticmethod
    def ttc(distance_m: float | None, velocity_mps: float | None) -> float | None:
        if distance_m is None or velocity_mps is None or velocity_mps <= 0:
            return None
        return distance_m / velocity_mps

    def hazards(self, detections: list[Detection], hazards: list[Hazard]) -> list[Hazard]:
        result = list(hazards)
        for detection in detections:
            if detection.collision_risk < 0.15:
                continue
            result.append(Hazard(
                type="approaching_vehicle" if detection.class_name.lower() in {"car", "vehicle", "truck", "bus", "motorcycle"} and detection.motion_state == MotionState.APPROACHING else detection.class_name.lower(),
                confidence=detection.collision_risk,
                distance_m=detection.distance_m,
                direction=detection.direction,
                severity="high" if detection.collision_risk >= 0.65 else "medium",
                source=["risk_engine"],
                track_id=detection.track_id,
                path_relevance=detection.path_relevance,
            ))
        return result

    @staticmethod
    def sort_hazards(hazards: list[Hazard]) -> list[Hazard]:
        return sorted(hazards, key=lambda item: (-item.path_relevance, item.distance_m if item.distance_m is not None else 99.0, -item.confidence, item.severity))
