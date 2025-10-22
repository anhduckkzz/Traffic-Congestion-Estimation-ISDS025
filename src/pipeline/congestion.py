from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch

from src.models.detection import TrackEvent, YOLODetector
from src.models.segmentation.deeplab_seresnet import build_segmentation_model
from src.pipeline.roi import LaneROIExtractor, ROILine
from src.pipeline.speed import SpeedEstimator
from src.utils import get_logger
from src.utils.video import iter_video_frames


LEVEL_ORDER = ["A", "B", "C", "D", "E", "F"]


@dataclass
class LoSThresholds:
    density: Dict[str, float]
    speed_two_wheel: Dict[str, float]
    speed_four_wheel: Dict[str, float]
    default_vehicle_length_m: float = 3.0


@dataclass
class LaneMetrics:
    lane_index: int
    vehicle_count: int
    avg_speed_kph: float
    density: float
    level_of_service: str


@dataclass
class PipelineResult:
    window_start: float
    window_end: float
    lanes: List[LaneMetrics]


class CongestionClassifier:
    def __init__(self, thresholds: LoSThresholds) -> None:
        self.thresholds = thresholds

    def classify(self, density: float, avg_speed_kph: float, vehicle_type: str) -> str:
        density_level = self._level_from_density(density)
        speed_level = self._level_from_speed(avg_speed_kph, vehicle_type)
        return LEVEL_ORDER[max(LEVEL_ORDER.index(density_level), LEVEL_ORDER.index(speed_level))]

    def _level_from_density(self, density: float) -> str:
        for level in LEVEL_ORDER[:-1]:  # exclude F which is catch-all
            threshold = self.thresholds.density.get(level)
            if threshold is not None and density <= threshold:
                return level
        return "F"

    def _level_from_speed(self, speed_kph: float, vehicle_type: str) -> str:
        table = self.thresholds.speed_two_wheel if vehicle_type == "two_wheeler" else self.thresholds.speed_four_wheel
        for level in LEVEL_ORDER:
            min_speed = table.get(level)
            if min_speed is None:
                continue
            if speed_kph >= min_speed:
                return level
        return "F"


class WindowAggregator:
    def __init__(
        self,
        roi_lines: List[ROILine],
        meter_per_pixel: float,
        classifier: CongestionClassifier,
        window_seconds: float,
        min_objects: int,
        vehicle_type_map: Dict[int, str],
    ) -> None:
        self.roi_lines = roi_lines
        self.meter_per_pixel = meter_per_pixel
        self.classifier = classifier
        self.window_seconds = window_seconds
        self.min_objects = min_objects
        self.vehicle_type_map = vehicle_type_map
        self.reset(start_time=0.0)

    def reset(self, start_time: float) -> None:
        self.window_start = start_time
        self.counts = [0 for _ in self.roi_lines]
        self.speed_samples: List[List[float]] = [[] for _ in self.roi_lines]
        self.type_counts: List[Dict[str, int]] = [dict(two_wheeler=0, four_wheeler=0) for _ in self.roi_lines]
        self.last_timestamp = start_time

    def record(self, lane_idx: int, class_id: int, speed_kph: Optional[float], timestamp: float) -> Optional[PipelineResult]:
        vehicle_type = self.vehicle_type_map.get(class_id, "four_wheeler")
        self.counts[lane_idx] += 1
        self.type_counts[lane_idx][vehicle_type] = self.type_counts[lane_idx].get(vehicle_type, 0) + 1
        if speed_kph is not None and speed_kph > 0:
            self.speed_samples[lane_idx].append(speed_kph)
        self.last_timestamp = timestamp

        if timestamp - self.window_start >= self.window_seconds:
            if sum(self.counts) < self.min_objects:
                self.reset(timestamp)
                return None
            result = self._build_result(timestamp)
            self.reset(timestamp)
            return result
        return None

    def _build_result(self, window_end: float) -> PipelineResult:
        lanes: List[LaneMetrics] = []
        for idx, roi in enumerate(self.roi_lines):
            vehicle_count = self.counts[idx]
            speeds = self.speed_samples[idx]
            avg_speed = mean(speeds) if speeds else 0.0
            dominant_type = "two_wheeler" if self.type_counts[idx].get("two_wheeler", 0) >= self.type_counts[idx].get("four_wheeler", 0) else "four_wheeler"
            distance_km = max(avg_speed * (self.window_seconds / 3600.0), 1e-3)
            density = vehicle_count / distance_km if vehicle_count else 0.0
            los = self.classifier.classify(density=density, avg_speed_kph=avg_speed, vehicle_type=dominant_type)
            lanes.append(
                LaneMetrics(
                    lane_index=idx,
                    vehicle_count=vehicle_count,
                    avg_speed_kph=avg_speed,
                    density=density,
                    level_of_service=los,
                )
            )
        return PipelineResult(window_start=self.window_start, window_end=window_end, lanes=lanes)


class CongestionPipeline:
    def __init__(
        self,
        segmentation_weights: str | Path,
        detector_weights: str | Path,
        tracker_config: str | Path,
        roi_extractor: LaneROIExtractor,
        thresholds: LoSThresholds,
        device: str = "cuda",
        window_seconds: float = 30.0,
        min_objects: int = 5,
        vehicle_type_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.segmentation_model = build_segmentation_model(num_classes=2)
        checkpoint = torch.load(segmentation_weights, map_location=self.device)
        self.segmentation_model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        self.segmentation_model.to(self.device)
        self.segmentation_model.eval()

        self.detector = YOLODetector(weights=detector_weights, device=str(self.device))
        self.tracker_config = tracker_config
        self.roi_extractor = roi_extractor
        self.thresholds = thresholds
        self.classifier = CongestionClassifier(thresholds)
        self.window_seconds = window_seconds
        self.min_objects = min_objects
        self.vehicle_type_map = vehicle_type_map or {}

        self.roi_lines: List[ROILine] = []
        self.meter_per_pixel: float = 0.0
        self.speed_estimator: Optional[SpeedEstimator] = None
        self.track_states: Dict[int, Dict[str, float]] = {}
        self.track_speeds: Dict[int, float] = {}

    def _infer_segmentation_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.segmentation_model(tensor)["out"]
        if output.shape[1] == 1:
            mask = torch.sigmoid(output)[0, 0] > 0.5
        else:
            mask = output.argmax(dim=1)[0]
        return mask.cpu().numpy().astype(np.uint8)

    def _initialize_roi(self, video_path: str | Path) -> None:
        for _, _, frame in iter_video_frames(video_path):
            mask = self._infer_segmentation_mask(frame)
            self.roi_lines, self.meter_per_pixel = self.roi_extractor.extract(mask)
            if not self.roi_lines:
                raise RuntimeError("Failed to derive ROI lines from segmentation mask")
            self.speed_estimator = SpeedEstimator(self.meter_per_pixel)
            self.logger.info("Initialized %d ROI lines (meter/pixel=%.4f)", len(self.roi_lines), self.meter_per_pixel)
            break

    def process(self, video_path: str | Path) -> Iterable[PipelineResult]:
        if not self.roi_lines:
            self._initialize_roi(video_path)
        assert self.speed_estimator is not None

        aggregator = WindowAggregator(
            roi_lines=self.roi_lines,
            meter_per_pixel=self.meter_per_pixel,
            classifier=self.classifier,
            window_seconds=self.window_seconds,
            min_objects=self.min_objects,
            vehicle_type_map=self.vehicle_type_map,
        )

        for event in self.detector.track(source=video_path, tracker_config=self.tracker_config):
            if event.track_id == -1:
                continue
            speed_estimate = self.speed_estimator.update(event)
            if speed_estimate:
                self.track_speeds[event.track_id] = speed_estimate.speed_kph

            centroid_y = 0.5 * (event.bbox_xyxy[1] + event.bbox_xyxy[3])
            state = self.track_states.setdefault(event.track_id, {"last_y": centroid_y})
            last_y = state["last_y"]
            for idx, roi in enumerate(self.roi_lines):
                if last_y < roi.y <= centroid_y:
                    speed_value = self.track_speeds.get(event.track_id)
                    result = aggregator.record(idx, event.class_id, speed_value, event.timestamp)
                    if result:
                        yield result
            state["last_y"] = centroid_y

        if sum(aggregator.counts) >= aggregator.min_objects:
            pending = aggregator._build_result(aggregator.last_timestamp)
            if pending.lanes:
                yield pending
