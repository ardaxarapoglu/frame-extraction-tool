import cv2
import numpy as np
import os
from typing import List, Callable, Optional

from .models import ProjectConfig, TimeFrame, CropRegion
from .tracker import ROITracker
from .obstruction_detector import ObstructionDetector
from .frame_selector import select_frames
from .naming import generate_filename


class VideoProcessor:
    def __init__(self, config: ProjectConfig,
                 progress_callback: Optional[Callable[[int, str], None]] = None,
                 cancel_check: Optional[Callable[[], bool]] = None):
        self.config = config
        self.progress_callback = progress_callback or (lambda p, m: None)
        self.cancel_check = cancel_check or (lambda: False)

    def process_all(self):
        video_dir = self.config.video_directory
        if not os.path.isdir(video_dir):
            self.progress_callback(0, f"Video directory not found: {video_dir}")
            return

        extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
        video_files = [f for f in sorted(os.listdir(video_dir))
                       if f.lower().endswith(extensions)]

        if not video_files:
            self.progress_callback(0, "No video files found in directory.")
            return

        total = len(video_files)
        for i, vf in enumerate(video_files):
            if self.cancel_check():
                self.progress_callback(0, "Processing cancelled.")
                return

            video_path = os.path.join(video_dir, vf)
            pct = int((i / total) * 100)
            self.progress_callback(pct, f"Processing {vf} ({i + 1}/{total})...")

            try:
                self._process_video(video_path, vf)
            except Exception as e:
                self.progress_callback(pct, f"Error processing {vf}: {e}")

        self.progress_callback(100, f"Done. Processed {total} video(s).")

    def _process_video(self, video_path: str, video_filename: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.progress_callback(0, f"Cannot open {video_filename}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        # Determine start time for this video
        if self.config.per_video_start and video_filename in self.config.video_start_marks:
            start_ms = self.config.video_start_marks[video_filename]
        else:
            start_ms = self.config.experiment_start_ms

        current_ms = start_ms

        for tf_idx, tf in enumerate(self.config.time_frames):
            if self.cancel_check():
                break

            self.progress_callback(
                0, f"  {video_filename}: extracting '{tf.name}' "
                   f"({tf_idx + 1}/{len(self.config.time_frames)})")

            frames, timestamps = self._extract_clip_frames(
                cap, fps, current_ms, tf.duration_seconds)

            if not frames:
                self.progress_callback(0, f"  No frames extracted for '{tf.name}'")
                current_ms += tf.duration_seconds * 1000
                continue

            # Apply rotation and cropping with tracking
            cropped_frames = self._crop_and_track(frames)

            # Detect obstructions
            detector = ObstructionDetector(sensitivity=self.config.obstruction_sensitivity)
            good_indices = detector.filter_frames(cropped_frames)

            self.progress_callback(
                0, f"  '{tf.name}': {len(good_indices)}/{len(cropped_frames)} "
                   f"frames passed obstruction filter")

            # Select evenly spaced frames
            selected = select_frames(good_indices, tf.num_frames)

            # Save frames
            self._save_frames(cropped_frames, selected, timestamps,
                              video_filename, tf)

            current_ms += tf.duration_seconds * 1000

        cap.release()

    def _extract_clip_frames(self, cap: cv2.VideoCapture, fps: float,
                             start_ms: float, duration_s: float):
        frames = []
        timestamps = []
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        end_ms = start_ms + duration_s * 1000
        while True:
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms >= end_ms:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            timestamps.append(pos_ms)
        return frames, timestamps

    def _crop_and_track(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        crop = self.config.crop_region
        if crop is None:
            return frames

        cropped = []
        tracker = ROITracker()
        first_frame = frames[0]

        # Apply rotation to get the working frame
        if crop.rotation_angle != 0:
            first_rotated = self._rotate_frame(first_frame, crop.rotation_angle)
        else:
            first_rotated = first_frame

        bbox = (crop.x, crop.y, crop.w, crop.h)
        tracker.init(first_rotated, bbox)

        for i, frame in enumerate(frames):
            if crop.rotation_angle != 0:
                frame = self._rotate_frame(frame, crop.rotation_angle)

            if i == 0:
                current_bbox = bbox
            else:
                _, current_bbox = tracker.update(frame)

            x, y, w, h = current_bbox
            # Clamp to frame boundaries
            fh, fw = frame.shape[:2]
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = min(w, fw - x)
            h = min(h, fh - y)

            crop_img = frame[y:y + h, x:x + w]
            if crop_img.size > 0:
                cropped.append(crop_img)
            else:
                cropped.append(frame)

        return cropped

    def _rotate_frame(self, frame: np.ndarray, angle: float) -> np.ndarray:
        h, w = frame.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(frame, M, (new_w, new_h))

    def _save_frames(self, frames: List[np.ndarray], selected_indices: List[int],
                     timestamps: List[float], video_filename: str, tf: TimeFrame):
        out_dir = os.path.join(self.config.output_directory, tf.name)
        os.makedirs(out_dir, exist_ok=True)

        for save_idx, frame_idx in enumerate(selected_indices):
            if frame_idx < len(frames):
                ts = timestamps[frame_idx] if frame_idx < len(timestamps) else 0.0
                filename = generate_filename(
                    tf.naming_scheme, video_filename, tf.name, save_idx, ts)
                filepath = os.path.join(out_dir, filename)
                cv2.imwrite(filepath, frames[frame_idx])

        self.progress_callback(
            0, f"  Saved {len(selected_indices)} frames for '{tf.name}'")
