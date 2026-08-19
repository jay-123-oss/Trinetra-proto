"""Backend facade for the Colab runtime.

The actual AI modules live in ``colab.trinetra`` so they can also run directly
inside the existing notebook. This facade keeps the production-style backend
layout without creating a second model-loading path.
"""

from __future__ import annotations

from typing import Any

from colab.trinetra.config import Settings
from colab.trinetra.server import create_app
from colab.trinetra.runtime import TrinetraRuntime
from colab.trinetra.camera import MjpegCamera


def build_application(existing_yolo_model: Any, existing_vlm_model: Any, settings: Settings | None = None):
    runtime = TrinetraRuntime(existing_yolo_model, existing_vlm_model, settings=settings)
    camera = MjpegCamera(runtime.settings.camera_url, runtime.buffer, runtime.metrics) if runtime.settings.camera_url else None
    return create_app(runtime, camera)


__all__ = ["build_application", "create_app"]
