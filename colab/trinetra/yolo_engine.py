from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from .config import Settings
from .models import Detection, Frame

logger = logging.getLogger(__name__)


class ExistingYolo(Protocol):
    def predict(self, source: Any, **kwargs: Any) -> Any: ...


class YoloEngine:
    """Adapter around the YOLO object already loaded in the Colab notebook."""

    def __init__(self, existing_model: ExistingYolo | Callable[[Any], Any], settings: Settings) -> None:
        if existing_model is None:
            raise ValueError("existing_yolo_model is required")
        self.model = existing_model
        self.settings = settings
        self.loaded = True
        self.inference_calls = 0

    def warmup(self, sample_frame: Any | None = None) -> None:
        warmup = getattr(self.model, "warmup", None)
        if callable(warmup):
            try:
                warmup()
                logger.info("reused YOLO warmup method")
            except Exception:
                logger.warning("YOLO warmup failed; continuing with persistent model", exc_info=True)
        elif sample_frame is not None:
            try:
                self._predict(sample_frame)
            except Exception:
                logger.warning("YOLO sample warmup failed", exc_info=True)

    def infer(self, frame: Frame) -> tuple[list[Detection], float]:
        started = time.perf_counter()
        raw = self._predict(frame.image)
        self.inference_calls += 1
        detections = self.normalize(raw, timestamp=frame.captured_at)
        return detections, (time.perf_counter() - started) * 1000.0

    def _predict(self, image: Any) -> Any:
        predict = getattr(self.model, "predict", None)
        if callable(predict):
            return predict(image, conf=self.settings.yolo_confidence, iou=self.settings.yolo_iou, verbose=False)
        if callable(self.model):
            return self.model(image)
        raise TypeError("existing YOLO object must expose predict() or be callable")

    @classmethod
    def normalize(cls, raw: Any, *, timestamp: float) -> list[Detection]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            return cls._dict_results(raw, timestamp)
        if isinstance(raw, (list, tuple)):
            output: list[Detection] = []
            for item in raw:
                if isinstance(item, dict):
                    output.extend(cls._dict_results(item, timestamp))
                elif isinstance(item, Detection):
                    output.append(item)
            return output
        # Ultralytics Results: boxes.xyxy/conf/cls and names on the result.
        results = raw if isinstance(raw, (list, tuple)) else [raw]
        output = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = cls._to_list(getattr(boxes, "xyxy", None))
            conf = cls._to_list(getattr(boxes, "conf", None))
            classes = cls._to_list(getattr(boxes, "cls", None))
            names = getattr(result, "names", {}) or {}
            for index, bbox in enumerate(xyxy):
                if len(bbox) < 4:
                    continue
                score = float(conf[index]) if index < len(conf) else 0.0
                class_id = int(classes[index]) if index < len(classes) else None
                class_name = str(names.get(class_id, class_id if class_id is not None else "unknown"))
                x1, y1, x2, y2 = map(float, bbox[:4])
                output.append(Detection(
                    track_id=None,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    center_x=(x1 + x2) / 2.0,
                    center_y=(y1 + y2) / 2.0,
                    width=max(0.0, x2 - x1),
                    height=max(0.0, y2 - y1),
                    timestamp=timestamp,
                ))
        return output

    @staticmethod
    def _dict_results(item: dict[str, Any], timestamp: float) -> list[Detection]:
        objects = item.get("objects", item.get("detections", [item] if "class_name" in item or "label" in item else []))
        output = []
        for obj in objects:
            name = str(obj.get("class_name", obj.get("label", "unknown")))
            bbox_raw = obj.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, dict):
                bbox = (float(bbox_raw.get("x1", 0)), float(bbox_raw.get("y1", 0)), float(bbox_raw.get("x2", 0)), float(bbox_raw.get("y2", 0)))
            else:
                values = list(bbox_raw)
                bbox = tuple(float(value) for value in (values + [0, 0, 0, 0])[:4])
            x1, y1, x2, y2 = bbox
            center = obj.get("center", {})
            center_x = float(obj.get("center_x", center.get("x", (x1 + x2) / 2.0)))
            center_y = float(obj.get("center_y", center.get("y", (y1 + y2) / 2.0)))
            output.append(Detection(
                track_id=obj.get("track_id"),
                class_id=obj.get("class_id"),
                class_name=name,
                confidence=float(obj.get("confidence", obj.get("score", 0.0))),
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                width=float(obj.get("width", x2 - x1)),
                height=float(obj.get("height", y2 - y1)),
                timestamp=timestamp,
            ))
        return output

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            value = [value]
        return value
