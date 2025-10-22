import numpy as np
import cv2

from src.pipeline.roi import LaneROIExtractor


def test_roi_extraction_produces_lines():
    mask = np.zeros((120, 200), dtype=np.uint8)
    cv2.rectangle(mask, (40, 20), (50, 70), 255, -1)
    cv2.rectangle(mask, (150, 25), (160, 75), 255, -1)
    cv2.rectangle(mask, (42, 90), (52, 110), 255, -1)
    cv2.rectangle(mask, (148, 88), (158, 108), 255, -1)

    extractor = LaneROIExtractor(dash_lengths_m=(1.0,), y_group_tolerance_px=20, min_component_area=10)
    lines, meter_per_pixel = extractor.extract(mask)

    assert len(lines) >= 1
    assert meter_per_pixel > 0
    for line in lines:
        assert line.x_left < line.x_right
