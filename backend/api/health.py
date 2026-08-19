from typing import Any


def read_health(runtime) -> dict[str, Any]:
    return runtime.health()
