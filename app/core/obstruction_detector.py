import cv2
import numpy as np
from typing import List, Tuple, Optional


class ObstructionDetector:
    """Detect obstructions (hands, cards) entering/covering the crop region.

    Three complementary signals:

    1. Blue pixel ratio — counts HSV-blue pixels directly in the crop.
       A blue lab glove covering any significant portion of the crop is
       reliably caught regardless of texture or motion.
       This is the primary signal.

    2. Texture drop — Laplacian variance of the crop vs the clip's 80th
       percentile (not median, so corrupted frames don't lower the reference).
       Catches full-frame coverage by non-blue obstructions.

    3. Guard border motion — frame-to-frame abs diff on the ring of pixels
       just outside the crop. Catches entry/exit events.

    A frame is flagged if ANY signal exceeds its threshold.

    sensitivity (0–1): higher = more aggressive detection (catches more).
    """

    # HSV range for a typical blue lab glove (OpenCV H: 0-180)
    _BLUE_LOW  = np.array([95,  55, 40], dtype=np.uint8)
    _BLUE_HIGH = np.array([135, 255, 255], dtype=np.uint8)

    def __init__(self, sensitivity: float = 0.35, guard_margin: int = 40):
        self.sensitivity = sensitivity
        self.guard_margin = guard_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_frames(
        self,
        cropped_frames: List[np.ndarray],
        guard_regions: Optional[List[np.ndarray]] = None,
        cancel_check=None,
    ) -> List[int]:
        if not cropped_frames:
            return []
        cancel_check = cancel_check or (lambda: False)

        n = len(cropped_frames)
        has_guard = guard_regions is not None and len(guard_regions) == n

        # --- Pass 1: collect raw signals for the whole clip -------------
        crop_vars  = np.zeros(n)
        blue_ratios = np.zeros(n)
        guard_diffs = np.zeros(n)

        prev_guard_gray = (
            self._to_gray(guard_regions[0]) if has_guard else None)

        for i in range(n):
            if cancel_check():
                return list(range(n))

            crop_vars[i]   = self._laplacian_variance(cropped_frames[i])
            blue_ratios[i] = self._blue_pixel_ratio(cropped_frames[i])

            if has_guard and i > 0:
                cur = self._to_gray(guard_regions[i])
                guard_diffs[i] = float(
                    np.mean(cv2.absdiff(prev_guard_gray, cur)))
                prev_guard_gray = cur
            elif has_guard:
                prev_guard_gray = self._to_gray(guard_regions[0])

        if cancel_check():
            return list(range(n))

        # --- Pass 2: compute clip-level thresholds ----------------------

        # Use 80th percentile as the "clean froth" reference so that
        # obstructed frames don't pull down the reference value.
        crop_reference = float(np.percentile(crop_vars, 80))

        # Texture threshold: frame is bad if variance < (ratio * reference)
        # sensitivity 0.0  → ratio = 0.40  (mild — only very low variance flagged)
        # sensitivity 0.35 → ratio = 0.225 (default)
        # sensitivity 1.0  → ratio = 0.05  (aggressive — even slightly below normal)
        texture_ratio = 0.40 - self.sensitivity * 0.35
        crop_thresh = crop_reference * max(texture_ratio, 0.03)

        # Blue ratio threshold: frame is bad if blue_ratio > threshold
        # sensitivity 0.0  → 0.18  (only obvious full-hand coverage)
        # sensitivity 0.35 → 0.08  (default — significant partial coverage)
        # sensitivity 1.0  → 0.02  (even a small blue area flags the frame)
        blue_thresh = 0.18 - self.sensitivity * 0.16

        guard_thresh = None
        if has_guard:
            guard_median = float(np.median(guard_diffs[1:]) if n > 1 else 0)
            # Guard spike = diff > N× median guard movement
            # sensitivity 0.0  → 2.0×  (needs a big spike)
            # sensitivity 0.35 → 3.75× (default)
            # sensitivity 1.0  → 7.0×  (fires on small movements too)
            guard_mult = 2.0 + self.sensitivity * 5.0
            guard_thresh = guard_median * guard_mult

        # --- Pass 3: flag frames ----------------------------------------
        good_indices = []
        for i in range(n):
            if cancel_check():
                return list(range(n))

            texture_bad = crop_vars[i] < crop_thresh
            blue_bad    = blue_ratios[i] > blue_thresh
            guard_bad   = (
                has_guard and
                guard_thresh is not None and
                guard_diffs[i] > guard_thresh
            )

            if not (texture_bad or blue_bad or guard_bad):
                good_indices.append(i)

        return good_indices

    def score_frames(
        self,
        cropped_frames: List[np.ndarray],
        guard_regions: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[bool, float]]:
        """Return (is_obstructed, normalised_score) per frame."""
        if not cropped_frames:
            return []

        n = len(cropped_frames)
        has_guard = guard_regions is not None and len(guard_regions) == n

        crop_vars   = np.zeros(n)
        blue_ratios = np.zeros(n)
        guard_diffs = np.zeros(n)

        prev_guard_gray = (
            self._to_gray(guard_regions[0]) if has_guard else None)

        for i in range(n):
            crop_vars[i]   = self._laplacian_variance(cropped_frames[i])
            blue_ratios[i] = self._blue_pixel_ratio(cropped_frames[i])
            if has_guard and i > 0:
                cur = self._to_gray(guard_regions[i])
                guard_diffs[i] = float(
                    np.mean(cv2.absdiff(prev_guard_gray, cur)))
                prev_guard_gray = cur
            elif has_guard:
                prev_guard_gray = self._to_gray(guard_regions[0])

        crop_reference = float(np.percentile(crop_vars, 80))
        texture_ratio  = 0.40 - self.sensitivity * 0.35
        crop_thresh    = crop_reference * max(texture_ratio, 0.03)
        blue_thresh    = 0.18 - self.sensitivity * 0.16

        guard_thresh = None
        if has_guard:
            guard_median = float(np.median(guard_diffs[1:]) if n > 1 else 0)
            guard_mult   = 2.0 + self.sensitivity * 5.0
            guard_thresh = guard_median * guard_mult

        results = []
        for i in range(n):
            texture_bad = crop_vars[i] < crop_thresh
            blue_bad    = blue_ratios[i] > blue_thresh
            guard_bad   = (
                has_guard and
                guard_thresh is not None and
                guard_diffs[i] > guard_thresh
            )
            obstructed = texture_bad or blue_bad or guard_bad

            texture_score = crop_thresh / max(crop_vars[i], 1e-6)
            blue_score    = blue_ratios[i] / max(blue_thresh, 1e-6)
            guard_score   = (
                guard_diffs[i] / max(guard_thresh, 1e-6)
                if guard_thresh else 0.0)
            score = max(texture_score, blue_score, guard_score)
            results.append((obstructed, score))

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _laplacian_variance(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return float(np.var(lap))

    @classmethod
    def _blue_pixel_ratio(cls, frame: np.ndarray) -> float:
        """Fraction of crop pixels that are blue (lab-glove HSV range)."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cls._BLUE_LOW, cls._BLUE_HIGH)
        return float(np.count_nonzero(mask)) / mask.size

    # ------------------------------------------------------------------
    # Guard region extraction (called by video_processor)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_guard_region(
        full_frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        margin: int = 40,
    ) -> np.ndarray:
        """Ring of pixels surrounding the crop rectangle."""
        fh, fw = full_frame.shape[:2]
        margin = min(margin, x, y, fw - (x + w), fh - (y + h))
        margin = max(margin, 1)

        top    = full_frame[max(0, y - margin):y,          x:x + w]
        bottom = full_frame[y + h:min(fh, y + h + margin), x:x + w]
        left   = full_frame[y:y + h, max(0, x - margin):x]
        right  = full_frame[y:y + h, x + w:min(fw, x + w + margin)]

        target_w = max(w, 1)
        strips = []
        for strip in [top, bottom]:
            if strip.size > 0:
                if strip.shape[1] != target_w:
                    strip = cv2.resize(strip, (target_w, strip.shape[0]))
                strips.append(strip)
        for strip in [left, right]:
            if strip.size > 0:
                strips.append(cv2.resize(strip, (target_w, strip.shape[0])))

        return np.vstack(strips) if strips else np.zeros(
            (1, target_w, 3), dtype=np.uint8)
