from __future__ import annotations

from .models import Hazard, VLMResult


class FusionEngine:
    """Combines evidence without allowing semantic VLM output to erase physical risk."""

    def fuse(self, hazards: list[Hazard], vlm: VLMResult) -> list[Hazard]:
        fused = list(hazards)
        if vlm.valid and vlm.hazard and vlm.hazard_type:
            matching = next((item for item in fused if item.type == vlm.hazard_type), None)
            if matching:
                matching.confidence = min(1.0, 0.65 * matching.confidence + 0.35 * vlm.confidence)
                matching.source = sorted(set(matching.source + ["vlm"]))
            else:
                fused.append(Hazard(
                    type=vlm.hazard_type,
                    confidence=vlm.confidence * 0.75,
                    distance_m=None,
                    direction=vlm.direction,
                    severity=vlm.severity,
                    source=["vlm"],
                    path_relevance=0.65,
                ))
        return fused

    @staticmethod
    def choose(hazards: list[Hazard]) -> Hazard | None:
        if not hazards:
            return None
        severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return max(hazards, key=lambda item: (item.path_relevance, -(item.distance_m if item.distance_m is not None else 99.0), severity_rank.get(item.severity, 0), item.confidence))
