from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from .camera import MjpegCamera
from .runtime import TrinetraRuntime


def create_app(runtime: TrinetraRuntime, camera: MjpegCamera | None = None) -> FastAPI:
    runtime.attach_camera(camera) if camera else None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(title="Trinetra Vision AI", version="1.0.0", lifespan=lifespan)

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

    return app
