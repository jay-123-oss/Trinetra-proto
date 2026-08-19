from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MotionState(str, Enum):
    STATIONARY = "STATIONARY"
    APPROACHING = "APPROACHING"
    MOVING_AWAY = "MOVING_AWAY"
    MOVING_LEFT = "MOVING_LEFT"
    MOVING_RIGHT = "MOVING_RIGHT"
    CROSSING = "CROSSING"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    DANGER = "DANGER"
    EMERGENCY = "EMERGENCY"


@dataclass(slots=True)
class Frame:
    frame_id: int
    image: Any
    captured_at: float
    width: int
    height: int
    orientation: str


@dataclass(slots=True)
class FrameQuality:
    valid: bool = True
    brightness: float = 0.0
    contrast: float = 0.0
    blur: float = 0.0
    visibility: float = 1.0
    error: str | None = None


@dataclass(slots=True)
class Detection:
    track_id: int | None
    class_id: int | None
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    width: float
    height: float
    timestamp: float
    distance_m: float | None = None
    distance_confidence: float = 0.0
    distance_source: str = "unknown"
    direction: str = "unknown"
    path_overlap: float = 0.0
    path_relevance: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_mps: float | None = None
    motion_state: MotionState = MotionState.UNKNOWN
    ttc_s: float | None = None
    collision_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["motion_state"] = self.motion_state.value
        data["bbox"] = {"x1": self.bbox[0], "y1": self.bbox[1], "x2": self.bbox[2], "y2": self.bbox[3]}
        data["center"] = {"x": self.center_x, "y": self.center_y}
        data.pop("center_x", None)
        data.pop("center_y", None)
        return data


@dataclass(slots=True)
class SpatialResult:
    distance_m: float | None = None
    confidence: float = 0.0
    direction: str = "unknown"
    position_x: float = 0.0
    position_y: float = 0.0
    source: str = "unknown"
    zone: str = "UNKNOWN"
    path_overlap: float = 0.0
    path_relevance: float = 0.0
    depth_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Hazard:
    type: str
    confidence: float
    distance_m: float | None
    direction: str
    severity: str
    source: list[str] = field(default_factory=list)
    track_id: int | None = None
    path_relevance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VLMResult:
    used: bool = False
    valid: bool = False
    hazard: bool = False
    hazard_type: str | None = None
    confidence: float = 0.0
    direction: str = "unknown"
    severity: str = "unknown"
    recommended_action: str = "unknown"
    latency_ms: float = 0.0
    error: str | None = None
    frame_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SafetyState:
    risk_level: RiskLevel = RiskLevel.SAFE
    hazard: str | None = None
    confidence: float = 0.0
    distance_m: float | None = None
    distance_confidence: float = 0.0
    direction: str = "unknown"
    severity: str = "low"
    recommended_action: str = "continue"
    message: str = "Path clear."
    reason: str = "no relevant hazard"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["risk_level"] = self.risk_level.value
        data["danger"] = self.risk_level in {RiskLevel.DANGER, RiskLevel.EMERGENCY}
        return data


@dataclass(slots=True)
class AlertEvent:
    should_alert: bool
    priority: RiskLevel
    message: str
    hazard: str | None = None
    track_id: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["priority"] = self.priority.value
        return data


@dataclass(slots=True)
class FrameAnalysis:
    timestamp: str
    frame_id: int
    frame_quality: FrameQuality = field(default_factory=FrameQuality)
    detections: list[Detection] = field(default_factory=list)
    tracking: dict[str, Any] = field(default_factory=dict)
    motion: dict[str, Any] = field(default_factory=dict)
    spatial: dict[str, Any] = field(default_factory=dict)
    hazards: list[Hazard] = field(default_factory=list)
    vlm: VLMResult = field(default_factory=VLMResult)
    safety: SafetyState = field(default_factory=SafetyState)
    alert: AlertEvent = field(default_factory=lambda: AlertEvent(False, RiskLevel.SAFE, "Path clear."))
    performance: dict[str, float] = field(default_factory=dict)
    system_state: str = "INITIALIZING"
    error: str | None = None

    @classmethod
    def empty(cls, *, error: str | None = None) -> "FrameAnalysis":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_id=0,
            system_state="DEGRADED" if error else "INITIALIZING",
            error=error,
        )

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "risk_level": self.safety.risk_level.value,
            "danger": self.safety.risk_level in {RiskLevel.DANGER, RiskLevel.EMERGENCY},
            "hazard": self.safety.hazard,
            "hazard_confidence": round(self.safety.confidence, 3),
            "distance_m": self.safety.distance_m,
            "distance_confidence": round(self.safety.distance_confidence, 3),
            "direction": self.safety.direction,
            "severity": self.safety.severity,
            "recommended_action": self.safety.recommended_action,
            "message": self.safety.message,
            "objects": [item.to_dict() for item in self.detections],
            "hazards": [item.to_dict() for item in self.hazards],
            "yolo": {
                "latency_ms": round(self.performance.get("yolo_latency_ms", 0.0), 2),
                "fps": round(self.performance.get("processing_fps", 0.0), 2),
            },
            "tracking": self.tracking,
            "motion": self.motion,
            "spatial": self.spatial,
            "vlm": self.vlm.to_dict(),
            "alert": self.alert.to_dict(),
            "performance": {key: round(value, 2) for key, value in self.performance.items()},
            "system_state": self.system_state,
            "frame_id": self.frame_id,
            "error": self.error,
        }
