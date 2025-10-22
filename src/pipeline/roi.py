from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ROILine:
    y: float
    x_left: float
    x_right: float
    pixel_length: float

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x_left + self.x_right), self.y)

    @property
    def width(self) -> float:
        return abs(self.x_right - self.x_left)


class LaneROIExtractor:
    def __init__(
        self,
        dash_lengths_m: Sequence[float] = (1.0, 2.0),
        y_group_tolerance_px: int = 30,
        min_component_area: int = 40,
        aspect_ratio_range: Tuple[float, float] = (1.0, 15.0),
    ) -> None:
        self.dash_lengths_m = dash_lengths_m
        self.y_group_tolerance_px = y_group_tolerance_px
        self.min_component_area = min_component_area
        self.aspect_ratio_range = aspect_ratio_range

    def extract(self, mask: np.ndarray) -> Tuple[List[ROILine], float]:
        if mask.ndim != 2:
            raise ValueError("Segmentation mask must be a 2D array")

        binary = (mask > 0).astype(np.uint8) * 255
        binary = cv2.medianBlur(binary, 3)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components = []
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_component_area:
                continue
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            if width == 0 or height == 0:
                continue
            aspect = max(width, height) / max(min(width, height), 1)
            if not (self.aspect_ratio_range[0] <= aspect <= self.aspect_ratio_range[1]):
                continue
            cx, cy = centroids[i]
            components.append(
                {
                    "centroid": (float(cx), float(cy)),
                    "bbox": stats[i],
                    "length_px": float(max(width, height)),
                }
            )

        if not components:
            return [], 0.0

        groups: List[List[dict]] = []
        for comp in sorted(components, key=lambda c: c["centroid"][1]):  # sort by y
            cy = comp["centroid"][1]
            placed = False
            for group in groups:
                mean_y = np.mean([c["centroid"][1] for c in group])
                if abs(mean_y - cy) <= self.y_group_tolerance_px:
                    group.append(comp)
                    placed = True
                    break
            if not placed:
                groups.append([comp])

        roi_lines: List[ROILine] = []
        length_candidates: List[float] = []
        for group in groups:
            if len(group) < 2:
                continue
            group_sorted = sorted(group, key=lambda c: c["centroid"][0])
            left = group_sorted[0]
            right = group_sorted[-1]
            y = float(np.mean([item["centroid"][1] for item in group]))
            roi_lines.append(
                ROILine(
                    y=y,
                    x_left=float(left["centroid"][0]),
                    x_right=float(right["centroid"][0]),
                    pixel_length=float(abs(right["centroid"][0] - left["centroid"][0])),
                )
            )
            length_candidates.extend(item["length_px"] for item in group)

        if not roi_lines:
            return [], 0.0

        ratios = []
        for px_len in length_candidates:
            if px_len <= 0:
                continue
            for real_len in self.dash_lengths_m:
                ratios.append(real_len / px_len)
        meter_per_pixel = float(np.median(ratios)) if ratios else 0.0

        roi_lines.sort(key=lambda r: r.y)
        return roi_lines, meter_per_pixel
