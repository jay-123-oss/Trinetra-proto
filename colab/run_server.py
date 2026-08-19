"""Start Trinetra around models already loaded by the current Colab notebook."""

from __future__ import annotations

import logging
from typing import Any

from colab.trinetra.camera import MjpegCamera
from colab.trinetra.runtime import TrinetraRuntime
from colab.trinetra.server import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def build_app(*, existing_yolo_model: Any, existing_vlm_model: Any, camera_url: str | None = None, settings=None):
    """Use the current notebook's loaded model objects; never load duplicate weights."""
    runtime = TrinetraRuntime(existing_yolo_model, existing_vlm_model, settings=settings)
    camera = MjpegCamera(camera_url, runtime.buffer, runtime.metrics) if camera_url else None
    return create_app(runtime, camera)


if __name__ == "__main__":
    raise SystemExit("Import build_app from the existing Colab notebook after YOLO/VLM are already loaded.")
