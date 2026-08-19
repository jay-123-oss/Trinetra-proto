from __future__ import annotations

import importlib
import os

import uvicorn

from backend.main import build_application
from colab.trinetra.config import Settings


def load_provider():
    target = os.getenv("TRINETRA_MODEL_PROVIDER", "")
    if not target or ":" not in target:
        raise RuntimeError("Set TRINETRA_MODEL_PROVIDER=module:function to return (existing_yolo_model, existing_vlm_model) before starting.")
    module_name, function_name = target.split(":", 1)
    provider = getattr(importlib.import_module(module_name), function_name)
    return provider()


def main() -> None:
    settings = Settings.from_env()
    yolo_model, vlm_model = load_provider()
    app = build_application(yolo_model, vlm_model, settings)
    print(f"TRINETRA API: http://{settings.api_host}:{settings.api_port}")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
