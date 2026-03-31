import cv2
import numpy as np
from typing import List, Tuple


class ObstructionDetector:
    def __init__(self, sensitivity: float = 0.35, edge_threshold_ratio: float = 0.3,
                 reference_frame_count: int = 5):
        self.sensitivity = sensitivity
        self.edge_threshold_ratio = edge_threshold_ratio
        self.reference_frame_count = reference_frame_count
        self.reference_hist = None
        self.baseline_edge_density = None

    def build_reference(self, frames: List[np.ndarray]):
        if not frames:
            return
        hists = []
        edge_densities = []
        for frame in frames[:self.reference_frame_count]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                                [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            hists.append(hist)

            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
            edge_densities.append(np.mean(edges > 0))

        self.reference_hist = np.mean(hists, axis=0).astype(np.float32)
        self.baseline_edge_density = np.mean(edge_densities)

    def is_obstructed(self, frame: np.ndarray) -> Tuple[bool, float]:
        if self.reference_hist is None:
            return False, 0.0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        color_dist = cv2.compareHist(self.reference_hist, hist,
                                     cv2.HISTCMP_BHATTACHARYYA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size
        low_edge_cells = 0
        total_cells = 0
        for gy in range(grid_size):
            for gx in range(grid_size):
                cell = edges[gy * cell_h:(gy + 1) * cell_h,
                             gx * cell_w:(gx + 1) * cell_w]
                cell_density = np.mean(cell > 0)
                total_cells += 1
                if self.baseline_edge_density > 0:
                    if cell_density < self.baseline_edge_density * self.edge_threshold_ratio:
                        low_edge_cells += 1

        edge_anomaly = low_edge_cells / max(total_cells, 1)

        obstructed = (color_dist > self.sensitivity) or (edge_anomaly > 0.25)
        score = max(color_dist, edge_anomaly)
        return obstructed, score

    def filter_frames(self, frames: List[np.ndarray],
                      cancel_check=None) -> List[int]:
        if not frames:
            return []
        cancel_check = cancel_check or (lambda: False)
        self.build_reference(frames)
        good_indices = []
        for i, frame in enumerate(frames):
            if cancel_check():
                break
            is_bad, _ = self.is_obstructed(frame)
            if not is_bad:
                good_indices.append(i)
        return good_indices
