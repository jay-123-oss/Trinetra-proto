from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    camera_url: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    input_width: int = 640
    input_height: int = 640
    target_processing_fps: float = 15.0
    vlm_min_interval_ms: float = 2_000.0
    vlm_confidence_trigger: float = 0.60
    emergency_distance_m: float = 0.55
    danger_distance_m: float = 0.90
    warning_distance_m: float = 1.80
    caution_distance_m: float = 3.50
    alert_cooldown_s: float = 2.0
    track_timeout_s: float = 1.0
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        def boolean(name: str, default: bool) -> bool:
            return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}

        return cls(
            camera_url=os.getenv("TRINETRA_CAMERA_URL", ""),
            api_host=os.getenv("TRINETRA_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("TRINETRA_API_PORT", "8000")),
            yolo_confidence=float(os.getenv("TRINETRA_YOLO_CONFIDENCE", "0.25")),
            yolo_iou=float(os.getenv("TRINETRA_YOLO_IOU", "0.45")),
            input_width=int(os.getenv("TRINETRA_INPUT_WIDTH", "640")),
            input_height=int(os.getenv("TRINETRA_INPUT_HEIGHT", "640")),
            target_processing_fps=float(os.getenv("TRINETRA_PROCESSING_FPS", "15")),
            vlm_min_interval_ms=float(os.getenv("TRINETRA_VLM_MIN_INTERVAL_MS", "2000")),
            vlm_confidence_trigger=float(os.getenv("TRINETRA_VLM_CONFIDENCE_TRIGGER", "0.60")),
            emergency_distance_m=float(os.getenv("TRINETRA_EMERGENCY_DISTANCE_M", "0.55")),
            danger_distance_m=float(os.getenv("TRINETRA_DANGER_DISTANCE_M", "0.90")),
            warning_distance_m=float(os.getenv("TRINETRA_WARNING_DISTANCE_M", "1.80")),
            caution_distance_m=float(os.getenv("TRINETRA_CAUTION_DISTANCE_M", "3.50")),
            alert_cooldown_s=float(os.getenv("TRINETRA_ALERT_COOLDOWN_S", "2.0")),
            track_timeout_s=float(os.getenv("TRINETRA_TRACK_TIMEOUT_S", "1.0")),
            debug=boolean("TRINETRA_DEBUG", False),
        )
