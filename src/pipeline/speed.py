from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src.models.detection.yolo_wrapper import TrackEvent


@dataclass
class SpeedEstimate:
    track_id: int
    timestamp: float
    speed_kph: float
    displacement_m: float
    duration_s: float


@dataclass
class _TrackHistory:
    first_timestamp: float
    first_centroid: np.ndarray
    last_centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))


class SpeedEstimator:
    def __init__(self, meter_per_pixel: float) -> None:
        self.meter_per_pixel = meter_per_pixel
        self._histories: Dict[int, _TrackHistory] = {}

    def reset(self) -> None:
        self._histories.clear()

    def update(self, event: TrackEvent) -> Optional[SpeedEstimate]:
        if self.meter_per_pixel <= 0:
            return None

        x1, y1, x2, y2 = event.bbox_xyxy
        centroid = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)

        history = self._histories.get(event.track_id)
        if history is None:
            self._histories[event.track_id] = _TrackHistory(
                first_timestamp=event.timestamp,
                first_centroid=centroid,
                last_centroid=centroid,
            )
            return None

        duration = event.timestamp - history.first_timestamp
        if duration <= 0:
            return None

        displacement_px = float(np.linalg.norm(centroid - history.first_centroid))
        displacement_m = displacement_px * self.meter_per_pixel
        speed_mps = displacement_m / duration
        speed_kph = speed_mps * 3.6
        history.last_centroid = centroid

        return SpeedEstimate(
            track_id=event.track_id,
            timestamp=event.timestamp,
            speed_kph=float(speed_kph),
            displacement_m=displacement_m,
            duration_s=duration,
        )
