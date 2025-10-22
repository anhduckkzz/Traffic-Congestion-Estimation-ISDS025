from src.pipeline.congestion import (
    CongestionClassifier,
    LoSThresholds,
    PipelineResult,
    WindowAggregator,
)
from src.pipeline.roi import ROILine


def test_congestion_classifier_levels():
    thresholds = LoSThresholds(
        density={"A": 5, "B": 10, "C": 14, "D": 24, "E": 40},
        speed_two_wheel={"A": 40, "B": 35, "C": 30, "D": 25, "E": 20, "F": 0},
        speed_four_wheel={"A": 45, "B": 40, "C": 35, "D": 25, "E": 15, "F": 0},
    )
    classifier = CongestionClassifier(thresholds)
    assert classifier.classify(density=4, avg_speed_kph=50, vehicle_type="four_wheeler") == "A"
    assert classifier.classify(density=30, avg_speed_kph=10, vehicle_type="two_wheeler") == "F"


def test_window_aggregator_outputs_results():
    thresholds = LoSThresholds(
        density={"A": 5, "B": 10, "C": 14, "D": 24, "E": 40},
        speed_two_wheel={"A": 40, "B": 35, "C": 30, "D": 25, "E": 20, "F": 0},
        speed_four_wheel={"A": 45, "B": 40, "C": 35, "D": 25, "E": 15, "F": 0},
    )
    classifier = CongestionClassifier(thresholds)
    roi_lines = [ROILine(y=50.0, x_left=0.0, x_right=100.0, pixel_length=100.0)]
    aggregator = WindowAggregator(
        roi_lines=roi_lines,
        meter_per_pixel=0.01,
        classifier=classifier,
        window_seconds=30,
        min_objects=1,
        vehicle_type_map={0: "two_wheeler"},
    )

    result = aggregator.record(0, class_id=0, speed_kph=42.0, timestamp=31.0)
    assert isinstance(result, PipelineResult)
    lane = result.lanes[0]
    assert lane.vehicle_count == 1
    assert lane.level_of_service in {"A", "B", "C", "D", "E", "F"}
