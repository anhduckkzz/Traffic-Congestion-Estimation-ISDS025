from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from rich.console import Console

from src.config import load_config
from src.pipeline.congestion import CongestionPipeline, LoSThresholds
from src.pipeline.roi import LaneROIExtractor
from src.utils import configure_logging, get_logger

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the traffic congestion estimation pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("run_pipeline")

    cfg = load_config(args.config)

    inputs_cfg = cfg["inputs"]
    video_path = args.video or inputs_cfg.get("video_path")
    if video_path is None:
        raise ValueError("Video path must be provided via --video or config")

    roi_cfg = cfg["roi"]
    extractor = LaneROIExtractor(
        dash_lengths_m=roi_cfg.get("dash_lengths_m", (1.0, 2.0)),
        y_group_tolerance_px=roi_cfg.get("y_group_tolerance_px", 30),
        min_component_area=roi_cfg.get("min_component_area", 40),
        aspect_ratio_range=tuple(roi_cfg.get("aspect_ratio_range", (1.0, 15.0))),
    )

    thresholds_cfg = cfg["loss_thresholds"]
    thresholds = LoSThresholds(
        density=thresholds_cfg.get("density", {}),
        speed_two_wheel=thresholds_cfg.get("speed_two_wheel", {}),
        speed_four_wheel=thresholds_cfg.get("speed_four_wheel", {}),
        default_vehicle_length_m=thresholds_cfg.get("default_vehicle_length_m", 3.0),
    )

    reporting_cfg = cfg.get("reporting", {})
    vehicle_type_map = {int(k): v for k, v in reporting_cfg.get("class_map", {}).items()}

    pipeline = CongestionPipeline(
        segmentation_weights=cfg["segmentation"]["checkpoint"],
        detector_weights=cfg["tracking"]["detector_weights"],
        tracker_config=cfg["tracking"]["tracker_config"],
        roi_extractor=extractor,
        thresholds=thresholds,
        device=cfg["segmentation"].get("device", "cuda"),
        window_seconds=inputs_cfg.get("window_seconds", 30),
        min_objects=inputs_cfg.get("min_objects_per_window", 5),
        vehicle_type_map=vehicle_type_map,
    )

    results = list(pipeline.process(video_path))
    logger.info("Processed %d windows", len(results))

    for result in results:
        console.print(f"Window {result.window_start:.1f}-{result.window_end:.1f}s")
        for lane in result.lanes:
            console.print(
                f"  Lane {lane.lane_index}: count={lane.vehicle_count} speed={lane.avg_speed_kph:.1f} kph density={lane.density:.1f} LoS={lane.level_of_service}"
            )

    if reporting_cfg.get("export_json", False):
        output_dir = Path(inputs_cfg.get("output_dir", "outputs/pipeline"))
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "results.json"
        payload = [
            {
                "window_start": r.window_start,
                "window_end": r.window_end,
                "lanes": [
                    {
                        "lane_index": lane.lane_index,
                        "vehicle_count": lane.vehicle_count,
                        "avg_speed_kph": lane.avg_speed_kph,
                        "density": lane.density,
                        "level_of_service": lane.level_of_service,
                    }
                    for lane in r.lanes
                ],
            }
            for r in results
        ]
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Saved JSON report to {json_path}")


if __name__ == "__main__":
    main()

