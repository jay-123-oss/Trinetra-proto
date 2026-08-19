from __future__ import annotations

from typing import Any

from colab.trinetra.config import Settings
from colab.trinetra.runtime import TrinetraRuntime


def get_runtime(existing_yolo_model: Any, existing_vlm_model: Any, settings: Settings | None = None) -> TrinetraRuntime:
    """Create one process-level runtime around the already-loaded model objects."""
    return TrinetraRuntime(existing_yolo_model, existing_vlm_model, settings=settings)
