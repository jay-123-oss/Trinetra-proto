from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi.staticfiles import StaticFiles

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .camera import MjpegCamera
from .runtime import TrinetraRuntime
from .safety import SafetyConfig


def create_app(runtime: TrinetraRuntime, camera: MjpegCamera | None = None) -> FastAPI:
    if camera:
        runtime.attach_camera(camera)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="Trinetra Vision AI", version="1.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/analyze")
    async def analyze() -> dict[str, Any]:
        result = runtime.latest_analysis()
        if result.frame_id == 0 and camera and camera.error:
            raise HTTPException(status_code=503, detail="camera unavailable")
        return result.to_api_dict()

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return runtime.metrics_snapshot()

    @app.get("/events")
    async def events() -> dict[str, Any]:
        return {"events": runtime.events()}

    @app.post("/camera/start")
    async def camera_start() -> dict[str, Any]:
        if camera is None:
            raise HTTPException(status_code=400, detail="camera is not configured")
        camera.start()
        return {"status": "starting", "camera_status": camera.status, "url_configured": bool(camera.url)}

    @app.post("/camera/stop")
    async def camera_stop() -> dict[str, Any]:
        if camera is None:
            raise HTTPException(status_code=400, detail="camera is not configured")
        camera.stop()
        return {"status": "stopped", "camera_status": camera.status}

    @app.post("/camera/config")
    async def camera_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if camera is None:
            raise HTTPException(status_code=400, detail="camera is not configured")
        url = str(payload.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=422, detail="url must use http:// or https://")
        was_running = camera.connected
        camera.stop()
        camera.url = url
        if was_running:
            camera.start()
        return {"status": "configured", "camera_status": camera.status, "url_configured": True}

    @app.post("/settings")
    async def settings_update(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        allowed = {
            "yolo_confidence": float,
            "target_processing_fps": float,
            "vlm_min_interval_ms": float,
            "warning_distance_m": float,
            "danger_distance_m": float,
            "alert_cooldown_s": float,
            "debug": bool,
        }
        for key, caster in allowed.items():
            if key in payload:
                try:
                    setattr(runtime.settings, key, caster(payload[key]))
                except (TypeError, ValueError):
                    raise HTTPException(status_code=422, detail=f"invalid setting: {key}")
        runtime.safety.config = SafetyConfig.from_settings(runtime.settings)
        runtime.alerts.cooldown_s = runtime.settings.alert_cooldown_s
        return {"status": "saved", "settings": {key: getattr(runtime.settings, key) for key in allowed}}

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        if camera is None:
            raise HTTPException(status_code=404, detail="camera is not configured")

        async def frames():
            while True:
                jpeg = camera.latest_jpeg()
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.08)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def websocket_updates(websocket: WebSocket):
        await websocket.accept()
        last_event_signature = None
        try:
            while True:
                analysis = runtime.latest_analysis().to_api_dict()
                await websocket.send_json({
                    "type": "safety",
                    "risk_level": analysis["risk_level"],
                    "hazard": analysis["hazard"],
                    "distance_m": analysis["distance_m"],
                    "direction": analysis["direction"],
                    "action": analysis["recommended_action"],
                    "message": analysis["message"],
                    "system_state": analysis["system_state"],
                })
                metric = runtime.metrics_snapshot()
                await websocket.send_json({"type": "metrics", **metric})
                events = runtime.events()
                if events:
                    event = events[0]
                    signature = (event.get("timestamp"), event.get("message"))
                    if signature != last_event_signature:
                        await websocket.send_json({"type": "alert", **event})
                        last_event_signature = signature
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    return app
