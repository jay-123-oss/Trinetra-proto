from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .models import Detection, Hazard, RiskLevel, SafetyState


@dataclass(slots=True)
class SafetyConfig:
    emergency_distance_m: float
    danger_distance_m: float
    warning_distance_m: float
    caution_distance_m: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "SafetyConfig":
        return cls(settings.emergency_distance_m, settings.danger_distance_m, settings.warning_distance_m, settings.caution_distance_m)


class SafetyEngine:
    def __init__(self, settings: Settings) -> None:
        self.config = SafetyConfig.from_settings(settings)

    @staticmethod
    def degraded(reason: str, *, camera_error: bool = False) -> SafetyState:
        return SafetyState(
            risk_level=RiskLevel.CAUTION,
            hazard="camera_unavailable" if camera_error else "perception_unavailable",
            confidence=0.0,
            distance_m=None,
            distance_confidence=0.0,
            direction="unknown",
            severity="medium",
            recommended_action="be_careful",
            message="Camera unavailable. Proceed carefully." if camera_error else "Perception unavailable. Proceed carefully.",
            reason=reason,
        )

    def decide(self, detections: list[Detection], hazards: list[Hazard]) -> SafetyState:
        # The fast path is independent of VLM and cannot be downgraded later.
        immediate = [d for d in detections if d.path_relevance >= 0.55 and d.distance_m is not None and d.distance_m <= self.config.emergency_distance_m and (d.collision_risk >= 0.45 or d.confidence >= 0.70)]
        if immediate:
            item = max(immediate, key=lambda d: d.collision_risk)
            return self._from_detection(item, RiskLevel.EMERGENCY, "stop_now", "immediate collision path")
        danger = [d for d in detections if d.path_relevance >= 0.5 and d.distance_m is not None and d.distance_m <= self.config.danger_distance_m and (d.collision_risk >= 0.30 or d.confidence >= 0.65)]
        if danger:
            item = max(danger, key=lambda d: d.collision_risk)
            return self._from_detection(item, RiskLevel.DANGER, "stop", "close high-risk object in walking corridor")

        if not hazards:
            return SafetyState()
        selected = max(hazards, key=self._priority)
        distance = selected.distance_m
        if distance is not None and distance <= self.config.warning_distance_m:
            level = RiskLevel.WARNING
        elif distance is None and selected.path_relevance > 0.55:
            level = RiskLevel.WARNING
        elif distance is not None and distance <= self.config.caution_distance_m:
            level = RiskLevel.CAUTION
        elif selected.path_relevance > 0.7:
            level = RiskLevel.CAUTION
        else:
            level = RiskLevel.SAFE
        action = self._action(level, selected.direction)
        return SafetyState(
            risk_level=level,
            hazard=selected.type,
            confidence=selected.confidence,
            distance_m=distance,
            distance_confidence=0.8 if distance is not None else 0.0,
            direction=selected.direction,
            severity=selected.severity,
            recommended_action=action,
            message=self._message(level, selected),
            reason="highest-priority multimodal hazard",
        )

    @staticmethod
    def _priority(hazard: Hazard) -> tuple[float, float, float, float]:
        distance_priority = 0.0 if hazard.distance_m is None else -hazard.distance_m
        severity = {"low": 0.0, "medium": 1.0, "high": 2.0, "critical": 3.0}.get(hazard.severity, 0.0)
        return hazard.path_relevance, distance_priority, severity, hazard.confidence

    @staticmethod
    def _from_detection(item: Detection, level: RiskLevel, action: str, reason: str) -> SafetyState:
        return SafetyState(
            risk_level=level,
            hazard=item.class_name,
            confidence=item.confidence,
            distance_m=item.distance_m,
            distance_confidence=item.distance_confidence,
            direction=item.direction,
            severity="critical" if level == RiskLevel.EMERGENCY else "high",
            recommended_action=action,
            message=f"{action.replace('_', ' ').capitalize()}. {item.class_name} ahead.",
            reason=reason,
        )

    @staticmethod
    def _action(level: RiskLevel, direction: str) -> str:
        if level == RiskLevel.SAFE:
            return "continue"
        if level == RiskLevel.CAUTION:
            return "be_careful"
        if level == RiskLevel.WARNING:
            return "slow_down"
        if direction in {"left", "slight_left"}:
            return "avoid_right"
        if direction in {"right", "slight_right"}:
            return "avoid_left"
        return "stop"

    @staticmethod
    def _message(level: RiskLevel, hazard: Hazard) -> str:
        if level == RiskLevel.SAFE:
            return "Path clear."
        distance = "" if hazard.distance_m is None else f" approximately {hazard.distance_m:.1f} meters"
        return f"{level.value.title()}. {hazard.type.replace('_', ' ')} ahead{distance}."
