"""Utilities to load and execute the congestion pipeline for the demo app."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

from src.config import load_config
from src.pipeline.congestion import CongestionPipeline, LoSThresholds, PipelineResult
from src.pipeline.roi import LaneROIExtractor


class PipelineConfigurationError(RuntimeError):
    """Raised when required configuration values are missing."""


def _require(mapping: Mapping[str, Any], section: str, key: str) -> Any:
    if key not in mapping:
        raise PipelineConfigurationError(f"Missing '{key}' in [{section}] section of the pipeline config")
    return mapping[key]


def build_congestion_pipeline(
    config_path: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> tuple[CongestionPipeline, dict[str, Any]]:
    """
    Build a :class:`CongestionPipeline` instance from the YAML configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    overrides:
        Optional nested mapping that overrides values inside the configuration.
        This leverages the existing deep-merge semantics from ``load_config``.
    """

    cfg_path = Path(config_path)
    config = load_config(cfg_path, overrides=overrides)

    inputs_cfg: Mapping[str, Any] = config.get("inputs", {})
    segmentation_cfg: MutableMapping[str, Any] = dict(config.get("segmentation", {}))
    roi_cfg: Mapping[str, Any] = config.get("roi", {})
    tracking_cfg: MutableMapping[str, Any] = dict(config.get("tracking", {}))
    thresholds_cfg: Mapping[str, Any] = config.get("loss_thresholds", {})
    reporting_cfg: Mapping[str, Any] = config.get("reporting", {})

    checkpoint_path = _require(segmentation_cfg, "segmentation", "checkpoint")
    detector_weights = _require(tracking_cfg, "tracking", "detector_weights")
    tracker_config = _require(tracking_cfg, "tracking", "tracker_config")

    roi_extractor = LaneROIExtractor(
        dash_lengths_m=tuple(roi_cfg.get("dash_lengths_m", (1.0, 2.0))),
        y_group_tolerance_px=int(roi_cfg.get("y_group_tolerance_px", 30)),
        min_component_area=int(roi_cfg.get("min_component_area", 40)),
        aspect_ratio_range=tuple(roi_cfg.get("aspect_ratio_range", (0.2, 5.0))),
    )

    thresholds = LoSThresholds(
        density=dict(thresholds_cfg.get("density", {})),
        speed_two_wheel=dict(thresholds_cfg.get("speed_two_wheel", {})),
        speed_four_wheel=dict(thresholds_cfg.get("speed_four_wheel", {})),
        default_vehicle_length_m=float(thresholds_cfg.get("default_vehicle_length_m", 3.0)),
    )

    vehicle_type_map = {int(k): str(v) for k, v in reporting_cfg.get("class_map", {}).items()}

    pipeline = CongestionPipeline(
        segmentation_weights=checkpoint_path,
        detector_weights=detector_weights,
        tracker_config=tracker_config,
        roi_extractor=roi_extractor,
        thresholds=thresholds,
        device=segmentation_cfg.get("device", "cuda"),
        window_seconds=float(inputs_cfg.get("window_seconds", 30.0)),
        min_objects=int(inputs_cfg.get("min_objects_per_window", 5)),
        vehicle_type_map=vehicle_type_map,
    )

    if "conf_threshold" in tracking_cfg:
        pipeline.detector.conf_threshold = float(tracking_cfg["conf_threshold"])
    if "iou_threshold" in tracking_cfg:
        pipeline.detector.iou_threshold = float(tracking_cfg["iou_threshold"])

    return pipeline, config


def run_pipeline(
    video_path: str | Path,
    config_path: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> tuple[list[PipelineResult], dict[str, Any]]:
    """Execute the congestion pipeline for the provided video."""
    pipeline, config = build_congestion_pipeline(config_path=config_path, overrides=overrides)
    results = list(pipeline.process(video_path))
    return results, config


def results_to_records(results: Iterable[PipelineResult]) -> list[dict[str, Any]]:
    """Flatten pipeline results into table-friendly dictionaries."""
    records: list[dict[str, Any]] = []
    for result in results:
        for lane in result.lanes:
            records.append(
                {
                    "window_start_s": round(result.window_start, 2),
                    "window_end_s": round(result.window_end, 2),
                    "lane_index": lane.lane_index,
                    "vehicle_count": lane.vehicle_count,
                    "avg_speed_kph": round(lane.avg_speed_kph, 2),
                    "density": round(lane.density, 2),
                    "level_of_service": lane.level_of_service,
                }
            )
    return records


def summarize_by_lane(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-lane metrics across all time windows."""
    aggregates: dict[int, dict[str, Any]] = {}
    for row in records:
        lane_idx = int(row["lane_index"])
        bucket = aggregates.setdefault(
            lane_idx,
            {
                "lane_index": lane_idx,
                "window_count": 0,
                "vehicle_total": 0,
                "speed_samples": [],
                "density_samples": [],
                "los_counter": Counter(),
            },
        )
        bucket["window_count"] += 1
        bucket["vehicle_total"] += int(row["vehicle_count"])
        bucket["speed_samples"].append(float(row["avg_speed_kph"]))
        bucket["density_samples"].append(float(row["density"]))
        bucket["los_counter"][str(row["level_of_service"])] += 1

    summary: list[dict[str, Any]] = []
    for lane_idx, bucket in sorted(aggregates.items(), key=lambda item: item[0]):
        los_counter: Counter[str] = bucket["los_counter"]  # type: ignore[assignment]
        dominant_los = los_counter.most_common(1)[0][0] if los_counter else "N/A"
        speed_samples = bucket["speed_samples"]
        density_samples = bucket["density_samples"]
        summary.append(
            {
                "lane_index": lane_idx,
                "windows": bucket["window_count"],
                "total_vehicles": bucket["vehicle_total"],
                "avg_speed_kph": round(sum(speed_samples) / len(speed_samples), 2) if speed_samples else 0.0,
                "avg_density": round(sum(density_samples) / len(density_samples), 2) if density_samples else 0.0,
                "dominant_los": dominant_los,
            }
        )
    return summary


def results_to_json(results: Iterable[PipelineResult]) -> list[dict[str, Any]]:
    """Convert pipeline results into a JSON-serialisable payload."""
    payload: list[dict[str, Any]] = []
    for result in results:
        payload.append(
            {
                "window_start": float(result.window_start),
                "window_end": float(result.window_end),
                "lanes": [
                    {
                        "lane_index": lane.lane_index,
                        "vehicle_count": lane.vehicle_count,
                        "avg_speed_kph": float(lane.avg_speed_kph),
                        "density": float(lane.density),
                        "level_of_service": lane.level_of_service,
                    }
                    for lane in result.lanes
                ],
            }
        )
    return payload
