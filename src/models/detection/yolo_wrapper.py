from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError("ultralytics package is required for YOLODetector") from exc


@dataclass
class DetectionResult:
    frame_index: int
    timestamp: float
    boxes: np.ndarray  # shape [N, 4] in xyxy format
    scores: np.ndarray  # shape [N]
    class_ids: np.ndarray  # shape [N]


@dataclass
class TrackEvent:
    frame_index: int
    timestamp: float
    track_id: int
    class_id: int
    bbox_xyxy: np.ndarray
    is_confirmed: bool
    velocity: Optional[np.ndarray] = None


class YOLODetector:
    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model = YOLO(str(weights))
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def finetune(self, data_cfg: str | Path, epochs: int, batch: int, img_size: int, project_dir: str | Path, name: str, **kwargs) -> None:
        """Fine-tune the detector using Ultralytics training loop."""
        self.model.train(
            data=str(data_cfg),
            epochs=epochs,
            batch=batch,
            imgsz=img_size,
            device=self.device,
            project=str(project_dir),
            name=name,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            **kwargs,
        )

    def predict(self, frame: np.ndarray, frame_index: int, timestamp: float) -> DetectionResult:
        results = self.model.predict(
            source=frame,
            device=self.device,
            imgsz=frame.shape[0:2],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        boxes: List[np.ndarray] = []
        scores: List[float] = []
        class_ids: List[int] = []
        if results:
            prediction = results[0]
            if prediction.boxes is not None and len(prediction.boxes) > 0:
                boxes = prediction.boxes.xyxy.cpu().numpy()  # type: ignore[attr-defined]
                scores = prediction.boxes.conf.cpu().numpy()
                class_ids = prediction.boxes.cls.cpu().numpy().astype(int)
        return DetectionResult(
            frame_index=frame_index,
            timestamp=timestamp,
            boxes=np.asarray(boxes, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            class_ids=np.asarray(class_ids, dtype=np.int32),
        )

    def track(
        self,
        source: str | Path,
        tracker_config: str | Path,
        persist: bool = True,
        **kwargs,
    ) -> Generator[TrackEvent, None, None]:
        """Run BoT-SORT tracking via the built-in Ultralytics interface."""
        stream = self.model.track(
            source=str(source),
            device=self.device,
            tracker=str(tracker_config),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=persist,
            stream=True,
            **kwargs,
        )
        for result in stream:
            frame_idx = getattr(result, "frame_idx", 0)
            timestamp = getattr(result, "timestamp", frame_idx / max(result.fps or 1, 1))
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().numpy() if result.boxes.id is not None else np.full(len(boxes), -1)
            classes = result.boxes.cls.int().cpu().numpy() if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
            velocities = None
            if hasattr(result, "speed") and result.speed is not None:
                velocities = result.speed.cpu().numpy()
            for box, track_id, cls_id, vel in zip(boxes, ids, classes, velocities if velocities is not None else [None] * len(boxes)):
                yield TrackEvent(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    track_id=int(track_id),
                    class_id=int(cls_id),
                    bbox_xyxy=box.astype(float),
                    is_confirmed=track_id != -1,
                    velocity=None if vel is None else np.asarray(vel, dtype=float),
                )
