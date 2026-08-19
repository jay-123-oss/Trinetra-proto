from typing import Any


def read_analysis(runtime) -> dict[str, Any]:
    return runtime.latest_analysis().to_api_dict()
