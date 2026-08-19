from __future__ import annotations

import time

import numpy as np

from colab.trinetra.config import Settings
from colab.trinetra.runtime import TrinetraRuntime


class BenchmarkYolo:
    def predict(self, _image, **_kwargs):
        return [{"class_id": 0, "class_name": "obstacle", "confidence": 0.8, "bbox": [240, 180, 400, 600]}]


class BenchmarkVlm:
    def analyze(self, _image, _context):
        return {"hazard": False, "hazard_type": None, "confidence": 0.7, "direction": "center", "severity": "low", "recommended_action": "continue"}


def main(iterations: int = 30) -> None:
    runtime = TrinetraRuntime(BenchmarkYolo(), BenchmarkVlm(), Settings(vlm_min_interval_ms=10_000))
    runtime.start()
    try:
        start = time.perf_counter()
        for _ in range(iterations):
            runtime.submit_frame(np.zeros((640, 640, 3), dtype=np.uint8), width=640, height=640, orientation="square")
            time.sleep(0.005)
        time.sleep(0.2)
        elapsed = time.perf_counter() - start
        print({"iterations": iterations, "wall_seconds": round(elapsed, 3), "metrics": runtime.metrics_snapshot()})
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
