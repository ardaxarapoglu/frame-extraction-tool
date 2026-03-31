import cv2
import numpy as np
from typing import Tuple, Optional


class ROITracker:
    def __init__(self, max_failures: int = 30):
        self.tracker = None
        self.max_failures = max_failures
        self.consecutive_failures = 0
        self.last_good_bbox = None
        self.initial_bbox = None

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        self.initial_bbox = bbox
        self.last_good_bbox = bbox
        self.consecutive_failures = 0
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(frame, bbox)

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        if self.tracker is None:
            return False, self.initial_bbox

        success, bbox = self.tracker.update(frame)
        if success:
            bbox = tuple(int(v) for v in bbox)
            self.last_good_bbox = bbox
            self.consecutive_failures = 0
            return True, bbox
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self._reinit(frame)
            return False, self.last_good_bbox

    def _reinit(self, frame: np.ndarray):
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(frame, self.last_good_bbox)
        self.consecutive_failures = 0

    def get_current_bbox(self) -> Tuple[int, int, int, int]:
        return self.last_good_bbox or self.initial_bbox
