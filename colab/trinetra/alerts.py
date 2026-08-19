from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Settings
from .models import AlertEvent, RiskLevel, SafetyState


@dataclass(slots=True)
class AlertMemory:
    hazard: str | None = None
    track_id: int | None = None
    distance_m: float | None = None
    risk: RiskLevel = RiskLevel.SAFE
    last_alert_s: float = 0.0


class AlertManager:
    def __init__(self, settings: Settings) -> None:
        self.cooldown_s = settings.alert_cooldown_s
        self.memory = AlertMemory()

    def evaluate(self, state: SafetyState, *, track_id: int | None = None, now_s: float | None = None) -> AlertEvent:
        now = time.monotonic() if now_s is None else now_s
        current_rank = self._rank(state.risk_level)
        previous_rank = self._rank(self.memory.risk)
        risk_escalated = current_rank > previous_rank
        hazard_changed = state.hazard != self.memory.hazard or track_id != self.memory.track_id
        distance_decreased = state.distance_m is not None and self.memory.distance_m is not None and state.distance_m < self.memory.distance_m * 0.8
        cooldown_expired = now - self.memory.last_alert_s >= self.cooldown_s
        should_alert = risk_escalated or hazard_changed or distance_decreased or cooldown_expired
        if state.risk_level == RiskLevel.SAFE and not hazard_changed and not cooldown_expired:
            should_alert = False
        if should_alert:
            self.memory = AlertMemory(state.hazard, track_id, state.distance_m, state.risk_level, now)
        else:
            self.memory.distance_m = state.distance_m
            self.memory.risk = state.risk_level
        return AlertEvent(should_alert, state.risk_level, state.message, state.hazard, track_id)

    @staticmethod
    def _rank(level: RiskLevel) -> int:
        return {RiskLevel.SAFE: 0, RiskLevel.CAUTION: 1, RiskLevel.WARNING: 2, RiskLevel.DANGER: 3, RiskLevel.EMERGENCY: 4}[level]
