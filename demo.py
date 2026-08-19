from __future__ import annotations

import time

try:
    import cv2  # noqa: F401
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing demo dependency. Activate your venv and run: "
        "python -m pip install -r colab/requirements.txt"
    ) from exc

from colab.trinetra.config import Settings
from colab.trinetra.runtime import TrinetraRuntime


class DemoYolo:
    def predict(self, _image, **_kwargs):
        return [{
            "class_id": 0,
            "class_name": "obstacle",
            "confidence": 0.58,
            "bbox": [250, 220, 390, 600],
        }]


class DemoVlm:
    def analyze(self, _image, _context):
        return {
            "hazard": True,
            "hazard_type": "unknown_ground",
            "confidence": 0.82,
            "direction": "center",
            "severity": "medium",
            "recommended_action": "avoid",
        }


def main() -> None:
    settings = Settings(vlm_min_interval_ms=0.0)
    runtime = TrinetraRuntime(DemoYolo(), DemoVlm(), settings)
    runtime.start()
    try:
        for _ in range(3):
            frame_id = runtime.submit_frame(np.zeros((640, 640, 3), dtype=np.uint8), width=640, height=640, orientation="square")
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and runtime.latest_analysis().frame_id != frame_id:
                time.sleep(0.01)
            print(runtime.latest_analysis().to_api_dict())
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
