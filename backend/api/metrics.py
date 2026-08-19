from typing import Any


def read_metrics(runtime) -> dict[str, Any]:
    return runtime.metrics_snapshot()
