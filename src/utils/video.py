from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, Tuple

import cv2


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, float, any]]:
    """Yield (frame_index, timestamp_seconds, frame_bgr)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_idx / fps
            yield frame_idx, timestamp, frame
            frame_idx += 1
    finally:
        capture.release()


class VideoWriter:
    """Simple wrapper above OpenCV VideoWriter."""

    def __init__(self, path: str | Path, fps: float, frame_size: Tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to create video writer at {path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
