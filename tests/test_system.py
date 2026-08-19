from __future__ import annotations

import time

import numpy as np
import pytest

from colab.trinetra.alerts import AlertManager
from colab.trinetra.config import Settings
from colab.trinetra.frame_buffer import LatestFrameBuffer
from colab.trinetra.models import Detection, Frame, MotionState, RiskLevel
from colab.trinetra.risk import RiskEngine
from colab.trinetra.safety import SafetyEngine
from colab.trinetra.spatial import SpatialEngine
from colab.trinetra.tracker import Tracker
from colab.trinetra.vlm import VLMTriggerManager
from colab.trinetra.yolo_engine import YoloEngine


class FakeYolo:
    def predict(self, _image, **_kwargs):
        return [{"class_id": 0, "class_name": "person", "confidence": 0.91, "bbox": [10, 10, 100, 200]}]


class BrokenYolo:
    def predict(self, _image, **_kwargs):
        raise RuntimeError("yolo failed")


def detection(name="obstacle", confidence=0.9, distance=1.0, overlap=0.9):
    return Detection(None, None, name, confidence, (220, 100, 420, 600), 320, 350, 200, 500, 0.0, distance, 0.8, "object_size", "center", overlap, overlap, 0.0, 0.4, 1.0, MotionState.APPROACHING, None, 0.0)


def test_latest_frame_buffer_replaces_stale_frame():
    buffer = LatestFrameBuffer()
    buffer.publish("old")
    newest = buffer.publish("new")
    frame = buffer.take_latest(0.1)
    assert frame and frame.frame_id == newest and frame.image == "new"
    assert buffer.dropped_count == 1


def test_yolo_normalization_supports_dict_results():
    results = YoloEngine.normalize({"objects": [{"class_id": 2, "class_name": "car", "confidence": 0.8, "bbox": [1, 2, 11, 22]}]}, timestamp=1.0)
    assert len(results) == 1
    assert results[0].class_name == "car"
    assert results[0].center_x == 6.0


def test_tracker_keeps_id_for_overlapping_detection():
    tracker = Tracker(iou_threshold=0.1)
    first = tracker.update([detection()], now_s=1.0)
    second = tracker.update([detection()], now_s=1.1)
    assert first[0].track_id == second[0].track_id
    assert tracker.active_count == 1


def test_spatial_direction_and_corridor_are_normalized():
    frame = Frame(1, np.zeros((640, 640, 3), dtype=np.uint8), 1.0, 640, 640, "square")
    enriched, summary = SpatialEngine().enrich(frame, [detection()])
    assert enriched[0].direction == "center"
    assert enriched[0].path_relevance > 0.0
    assert summary["corridor_objects"] == 1


def test_ttc_is_safe_for_nonpositive_velocity_and_valid_for_approach():
    assert RiskEngine.ttc(1.0, 0.0) is None
    assert RiskEngine.ttc(1.0, 0.5) == 2.0


def test_immediate_collision_wins_without_vlm():
    state = SafetyEngine(Settings()).decide([detection(distance=0.4, overlap=1.0)], [])
    assert state.risk_level == RiskLevel.EMERGENCY
    assert state.recommended_action == "stop_now"


def test_vlm_json_validation_rejects_unknown_values():
    result = VLMTriggerManager.validate({"hazard": True, "hazard_type": "made_up", "confidence": 0.8, "direction": "center", "severity": "high"})
    assert result.valid is False
    result = VLMTriggerManager.validate({"hazard": True, "hazard_type": "pothole", "confidence": 0.8, "direction": "center", "severity": "high"})
    assert result.valid is True


def test_alert_manager_deduplicates_same_state_during_cooldown():
    manager = AlertManager(Settings(alert_cooldown_s=2.0))
    state = SafetyEngine(Settings()).decide([detection(distance=1.0)], [])
    first = manager.evaluate(state, now_s=10.0)
    second = manager.evaluate(state, now_s=10.5)
    assert first.should_alert is True
    assert second.should_alert is False


def test_api_health_uses_persistent_runtime_without_camera():
    from colab.trinetra.runtime import TrinetraRuntime
    runtime = TrinetraRuntime(FakeYolo(), None, Settings())
    health = runtime.health()
    assert health["yolo"] is True
    assert health["vlm"] is False
    runtime.stop()


def test_fastapi_health_and_analyze_endpoints():
    from fastapi.testclient import TestClient
    from colab.trinetra.runtime import TrinetraRuntime
    from colab.trinetra.server import create_app
    runtime = TrinetraRuntime(FakeYolo(), None, Settings())
    app = create_app(runtime)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["yolo"] is True
        analysis = client.get("/analyze")
        assert analysis.status_code == 200
        assert "risk_level" in analysis.json()


def test_dashboard_websocket_and_settings_endpoints():
    from fastapi.testclient import TestClient
    from colab.trinetra.runtime import TrinetraRuntime
    from colab.trinetra.server import create_app
    runtime = TrinetraRuntime(FakeYolo(), None, Settings())
    app = create_app(runtime)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "TRINETRA" in page.text
        settings = client.post("/settings", json={"warning_distance_m": 2.1, "debug": True})
        assert settings.status_code == 200
        assert settings.json()["status"] == "saved"
        with client.websocket_connect("/ws") as socket:
            update = socket.receive_json()
            assert update["type"] in {"safety", "metrics", "alert"}


def test_degraded_safety_is_not_false_safe():
    from colab.trinetra.safety import SafetyEngine
    state = SafetyEngine.degraded("yolo failed")
    assert state.risk_level == RiskLevel.CAUTION
    assert state.hazard == "perception_unavailable"
    assert state.recommended_action == "be_careful"


def test_invalid_frame_is_not_false_safe_in_runtime():
    from colab.trinetra.runtime import TrinetraRuntime
    runtime = TrinetraRuntime(FakeYolo(), None, Settings())
    runtime.start()
    try:
        frame_id = runtime.submit_frame(object())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and runtime.latest_analysis().frame_id != frame_id:
            time.sleep(0.01)
        analysis = runtime.latest_analysis()
        assert analysis.system_state == "DEGRADED"
        assert analysis.safety.risk_level == RiskLevel.CAUTION
    finally:
        runtime.stop()


def test_stale_vlm_result_is_not_returned_for_a_new_frame():
    manager = VLMTriggerManager(None, Settings())
    manager._latest.frame_id = 1
    assert manager.latest_for(10).used is False
    manager.stop()


def test_yolo_failure_is_catchable_at_engine_boundary():
    engine = YoloEngine(BrokenYolo(), Settings())
    frame = Frame(1, np.zeros((10, 10, 3), dtype=np.uint8), 1.0, 10, 10, "square")
    with pytest.raises(RuntimeError):
        engine.infer(frame)
