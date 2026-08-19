from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .alerts import AlertManager
from .config import Settings
from .frame_buffer import LatestFrameBuffer
from .fusion import FusionEngine
from .hazards import GroundHazardEngine, HazardEngine
from .metrics import Metrics
from .models import FrameAnalysis, RiskLevel, VLMResult
from .motion import MotionEngine
from .preprocess import FrameProcessor
from .risk import RiskEngine
from .safety import SafetyEngine
from .spatial import SpatialEngine
from .tracker import Tracker
from .vlm import VLMTriggerManager
from .yolo_engine import YoloEngine

logger = logging.getLogger(__name__)


class TrinetraRuntime:
    """One persistent Python runtime for existing YOLO and VLM instances."""

    def __init__(self, existing_yolo_model: Any, existing_vlm_model: Any, settings: Settings | None = None, *, depth_provider=None) -> None:
        self.settings = settings or Settings.from_env()
        self.buffer = LatestFrameBuffer()
        self.metrics = Metrics()
        self.processor = FrameProcessor(self.settings)
        self.yolo = YoloEngine(existing_yolo_model, self.settings)
        self.tracker = Tracker(timeout_s=self.settings.track_timeout_s)
        self.motion = MotionEngine()
        self.spatial = SpatialEngine(depth_provider)
        self.ground = GroundHazardEngine()
        self.hazards = HazardEngine()
        self.risk = RiskEngine()
        self.vlm = VLMTriggerManager(existing_vlm_model, self.settings)
        self.fusion = FusionEngine()
        self.safety = SafetyEngine(self.settings)
        self.alerts = AlertManager(self.settings)
        self._latest = FrameAnalysis.empty()
        self._result_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_processed_s: float | None = None
        self._system_state = "INITIALIZING"
        self.camera = None
        self._events = deque(maxlen=50)
        self._events_lock = threading.Lock()

    def attach_camera(self, camera) -> None:
        self.camera = camera

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self.yolo.warmup()
        self._worker = threading.Thread(target=self._run, name="trinetra-inference", daemon=True)
        self._worker.start()
        if self.camera:
            self.camera.start()
        self._system_state = "READY"
        self.metrics.system_state = "READY"
        logger.info("Trinetra runtime started; YOLO and VLM remain in the same Colab process")

    def stop(self) -> None:
        self._stop.set()
        if self.camera:
            self.camera.stop()
        self.buffer.close()
        self.vlm.stop()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def submit_frame(self, image: Any, *, captured_at: float | None = None, width: int | None = None, height: int | None = None, orientation: str = "unknown") -> int:
        return self.buffer.publish(image, captured_at=captured_at, width=width, height=height, orientation=orientation)

    def latest_analysis(self) -> FrameAnalysis:
        with self._result_lock:
            return self._latest

    def health(self) -> dict[str, Any]:
        gpu_available = self._gpu_available()
        camera_ok = self.camera is None or self.camera.connected
        yolo_ok = self.yolo.loaded
        vlm_ok = self.vlm.model is not None
        status = "healthy" if camera_ok and yolo_ok else "degraded"
        return {
            "status": status,
            "camera": camera_ok,
            "camera_status": self.camera.status if self.camera else "INJECTED_FRAMES",
            "yolo": yolo_ok,
            "vlm": vlm_ok,
            "gpu": gpu_available,
            "fps": self.metrics.snapshot(dropped_frames=self.buffer.dropped_count).get("processing_fps", 0.0),
            "system_state": self._system_state,
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot(dropped_frames=self.buffer.dropped_count, gpu_memory=self._gpu_memory())

    def events(self) -> list[dict[str, Any]]:
        with self._events_lock:
            return list(self._events)

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.buffer.take_latest()
            if frame is None:
                continue
            try:
                analysis = self._process(frame)
            except Exception as exc:
                logger.exception("inference loop frame failure")
                analysis = FrameAnalysis.empty(error=str(exc))
                analysis.frame_id = frame.frame_id
                analysis.system_state = "DEGRADED"
            with self._result_lock:
                self._latest = analysis

    def _process(self, frame) -> FrameAnalysis:
        total_start = time.perf_counter()
        processed, quality = self.processor.process(frame)
        if not quality.valid:
            self._system_state = "DEGRADED"
            return FrameAnalysis(datetime.now(timezone.utc).isoformat(), frame.frame_id, quality, system_state="DEGRADED", error=quality.error)

        stage: dict[str, float] = {}
        start = time.perf_counter()
        try:
            detections, yolo_latency = self.yolo.infer(processed)
        except Exception as exc:
            logger.exception("YOLO inference failed")
            detections, yolo_latency = [], 0.0
            yolo_error = str(exc)
            self._system_state = "YOLO_ERROR"
        else:
            yolo_error = None
        stage["yolo_latency_ms"] = yolo_latency
        tracked_start = time.perf_counter()
        tracked = self.tracker.update(detections)
        stage["tracking_latency_ms"] = (time.perf_counter() - tracked_start) * 1000.0
        motion_start = time.perf_counter()
        moving = self.motion.analyze(tracked)
        stage["motion_latency_ms"] = (time.perf_counter() - motion_start) * 1000.0
        spatial_start = time.perf_counter()
        spatial_detections, spatial_summary = self.spatial.enrich(processed, moving)
        stage["spatial_latency_ms"] = (time.perf_counter() - spatial_start) * 1000.0
        risk_start = time.perf_counter()
        risk_detections = self.risk.enrich_detections(spatial_detections)
        ground = self.ground.analyze(processed, quality)
        hazard_list = self.hazards.classify(risk_detections, ground, quality)
        hazard_list = self.risk.hazards(risk_detections, hazard_list)
        stage["risk_latency_ms"] = (time.perf_counter() - risk_start) * 1000.0
        preliminary = self.safety.decide(risk_detections, hazard_list)
        trigger = self.vlm.should_trigger(risk_detections, preliminary)
        if trigger:
            self.metrics.vlm_triggers += 1
            self.vlm.submit(processed, preliminary, risk_detections)
        vlm_result = self.vlm.latest()
        fused = self.fusion.fuse(hazard_list, vlm_result)
        final_safety = self.safety.decide(risk_detections, fused)
        alert = self.alerts.evaluate(final_safety, track_id=self._track_id(risk_detections, final_safety.hazard))
        now = time.monotonic()
        processing_fps = 0.0 if self._last_processed_s is None else 1.0 / max(1e-6, now - self._last_processed_s)
        self._last_processed_s = now
        stage["processing_fps"] = processing_fps
        stage["vlm_latency_ms"] = vlm_result.latency_ms
        stage["safety_latency_ms"] = 0.0
        stage["total_latency_ms"] = (time.perf_counter() - total_start) * 1000.0
        stage["network_latency_ms"] = max(0.0, (time.monotonic() - processed.captured_at) * 1000.0)
        self.metrics.record(stage)
        if alert.should_alert:
            with self._events_lock:
                self._events.appendleft(alert.to_dict())
        if vlm_result.used:
            self.metrics.vlm_calls = self.vlm.calls
        self._system_state = "DEGRADED" if yolo_error or vlm_result.error else final_safety.risk_level.value
        self.metrics.system_state = self._system_state
        logger.info("frame=%s state=%s hazard=%s latency_ms=%.1f", frame.frame_id, final_safety.risk_level.value, final_safety.hazard, stage["total_latency_ms"])
        return FrameAnalysis(
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_id=frame.frame_id,
            frame_quality=quality,
            detections=risk_detections,
            tracking={"active_tracks": self.tracker.active_count, "objects": [item.to_dict() for item in risk_detections]},
            motion={"objects": [{"track_id": item.track_id, "state": item.motion_state.value, "velocity_x": item.velocity_x, "velocity_y": item.velocity_y} for item in risk_detections]},
            spatial=spatial_summary,
            hazards=fused,
            vlm=vlm_result,
            safety=final_safety,
            alert=alert,
            performance=stage,
            system_state=self._system_state,
            error=yolo_error,
        )

    @staticmethod
    def _track_id(detections, hazard: str | None) -> int | None:
        for item in detections:
            if item.class_name.lower() == (hazard or "").lower():
                return item.track_id
        return None

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _gpu_memory() -> float | None:
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass
        return None
