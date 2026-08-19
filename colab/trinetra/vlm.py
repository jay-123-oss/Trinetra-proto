from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from typing import Any, Protocol

from .config import Settings
from .models import Detection, Frame, RiskLevel, SafetyState, VLMResult

logger = logging.getLogger(__name__)


class ExistingVlm(Protocol):
    def analyze(self, image: Any, context: dict[str, Any]) -> Any: ...


class VLMTriggerManager:
    """Non-blocking VLM bridge: one active call and at most one latest pending request."""

    VALID_TYPES = {"obstacle", "wall", "person", "vehicle", "approaching_vehicle", "motorcycle", "bicycle", "pothole", "stairs", "down_stairs", "road_edge", "open_drain", "uneven_ground", "barrier", "pole", "crowd", "blocked_path", "crosswalk", "unknown_hazard", "unknown_ground"}
    VALID_DIRECTIONS = {"left", "slight_left", "center", "slight_right", "right", "unknown"}
    VALID_SEVERITIES = {"low", "medium", "high", "critical", "unknown"}

    def __init__(self, existing_model: ExistingVlm | None, settings: Settings) -> None:
        self.model = existing_model
        self.settings = settings
        self._condition = threading.Condition()
        self._active = False
        self._pending: tuple[Frame, dict[str, Any]] | None = None
        self._latest = VLMResult()
        self._stop = False
        self.calls = 0
        self.failures = 0
        self.total_latency_ms = 0.0
        self._last_submit_ms = 0.0
        self._worker = threading.Thread(target=self._run, name="trinetra-vlm", daemon=True)
        self._worker.start()

    def should_trigger(self, detections: list[Detection], safety: SafetyState, *, scene_changed: bool = False) -> bool:
        if self.model is None:
            return False
        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_submit_ms < self.settings.vlm_min_interval_ms and safety.risk_level.value not in {"DANGER", "EMERGENCY"}:
            return False
        corridor = [item for item in detections if item.path_relevance > 0.35]
        low_confidence = any(item.confidence < self.settings.vlm_confidence_trigger for item in corridor)
        ambiguous = any(item.class_name.lower() in {"unknown", "obstacle", "pothole", "stairs", "road_edge", "open_drain"} for item in corridor)
        complex_scene = len(corridor) >= 3
        disagree = any(item.distance_confidence < 0.35 and item.path_relevance > 0.5 for item in corridor)
        return low_confidence or ambiguous or complex_scene or disagree or scene_changed or safety.risk_level in {RiskLevel.CAUTION, RiskLevel.WARNING}

    def submit(self, frame: Frame, safety: SafetyState, detections: list[Detection]) -> None:
        if self.model is None:
            return
        context = {
            "task": "Analyze this scene specifically for pedestrian safety. Return only valid JSON.",
            "safety_state": safety.to_dict(),
            "detections": [item.to_dict() for item in detections if item.path_relevance > 0.25],
            "frame_id": frame.frame_id,
            "image_policy": "use the latest relevant frame and crop/context when supported",
        }
        with self._condition:
            self._pending = (frame, context)
            self._last_submit_ms = time.monotonic() * 1000.0
            self._condition.notify()

    def latest(self) -> VLMResult:
        with self._condition:
            return self._latest

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait(timeout=0.25)
                if self._stop:
                    return
                request = self._pending
                self._pending = None
                self._active = True
            assert request is not None
            frame, context = request
            started = time.perf_counter()
            try:
                method = getattr(self.model, "analyze", None)
                raw = method(frame.image, context) if callable(method) else self.model(frame.image, context)  # type: ignore[misc]
                result = self.validate(raw)
                result.used = True
                result.frame_id = frame.frame_id
            except Exception as exc:
                self.failures += 1
                logger.exception("VLM request failed")
                result = VLMResult(used=True, valid=False, error=str(exc), frame_id=frame.frame_id)
            latency = (time.perf_counter() - started) * 1000.0
            result.latency_ms = latency
            self.calls += 1
            self.total_latency_ms += latency
            with self._condition:
                self._latest = result
                self._active = False

    @classmethod
    def validate(cls, raw: Any) -> VLMResult:
        if not isinstance(raw, dict):
            return VLMResult(valid=False, error="VLM output is not a JSON object")
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            return VLMResult(valid=False, error="invalid VLM confidence")
        hazard_type = raw.get("hazard_type")
        direction = raw.get("direction", "unknown")
        severity = raw.get("severity", "unknown")
        if not 0.0 <= confidence <= 1.0:
            return VLMResult(valid=False, error="confidence outside [0, 1]")
        if hazard_type is not None and hazard_type not in cls.VALID_TYPES:
            return VLMResult(valid=False, error="unknown hazard type")
        if direction not in cls.VALID_DIRECTIONS:
            return VLMResult(valid=False, error="invalid direction")
        if severity not in cls.VALID_SEVERITIES:
            return VLMResult(valid=False, error="invalid severity")
        return VLMResult(
            valid=True,
            hazard=bool(raw.get("hazard", bool(hazard_type))),
            hazard_type=hazard_type,
            confidence=confidence,
            direction=direction,
            severity=severity,
            recommended_action=str(raw.get("recommended_action", "unknown")),
        )
