from __future__ import annotations

from collections import Counter
from typing import Any

from .models import Detection, Frame, FrameQuality, Hazard


class GroundHazardEngine:
    """Conservative ground-region candidate generator; VLM verifies ambiguity."""

    def analyze(self, frame: Frame, quality: FrameQuality) -> list[Hazard]:
        if not quality.valid or quality.visibility < 0.25:
            return []
        try:
            import cv2
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            lower = gray[int(gray.shape[0] * 0.60):]
            if lower.size == 0:
                return []
            texture = float(cv2.Laplacian(lower, cv2.CV_64F).var())
            # Texture discontinuity is only a candidate signal, never a final alert.
            if texture < 12.0 and quality.blur > 0.35:
                return [Hazard("unknown_ground", 0.40, None, "center", "medium", ["ground_analysis"], path_relevance=0.7)]
        except Exception:
            return []
        return []


class HazardEngine:
    HIGH_IMPACT = {"vehicle", "car", "motorcycle", "bus", "truck", "bicycle", "wall", "barrier", "stairs", "down_stairs", "road_edge", "open_drain", "pothole"}
    NAME_MAP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "motorcycle", "bicycle": "bicycle", "person": "person"}

    def classify(self, detections: list[Detection], ground: list[Hazard], quality: FrameQuality) -> list[Hazard]:
        hazards: list[Hazard] = []
        for detection in detections:
            if detection.path_relevance < 0.2 and detection.motion_state.value != "APPROACHING":
                continue
            name = self.NAME_MAP.get(detection.class_name.lower(), detection.class_name.lower())
            severity = "high" if name in self.HIGH_IMPACT and detection.path_relevance > 0.5 else "medium" if detection.path_relevance > 0.4 else "low"
            confidence = detection.confidence * (0.55 + 0.45 * detection.path_relevance)
            if quality.visibility < 0.35:
                confidence *= 0.7
            hazards.append(Hazard(
                type=name,
                confidence=max(0.0, min(1.0, confidence)),
                distance_m=detection.distance_m,
                direction=detection.direction,
                severity=severity,
                source=["yolo", "tracking", "spatial"],
                track_id=detection.track_id,
                path_relevance=detection.path_relevance,
            ))
        hazards.extend(ground)
        people = sum(1 for d in detections if d.class_name.lower() == "person")
        if people >= 4:
            hazards.append(Hazard("crowd", min(1.0, people / 8.0), None, "center", "medium", ["tracking"], path_relevance=0.75))
        if quality.visibility < 0.3:
            hazards.append(Hazard("low_visibility", 1.0 - quality.visibility, None, "center", "medium", ["frame_quality"], path_relevance=0.5))
        return hazards

    @staticmethod
    def summary(hazards: list[Hazard]) -> dict[str, Any]:
        return {"count": len(hazards), "types": dict(Counter(item.type for item in hazards)), "highest_confidence": max((item.confidence for item in hazards), default=0.0)}
