import cv2
import numpy as np
import os
from typing import List, Callable, Optional, Tuple

from .models import ProjectConfig, TimeFrame, CropRegion
from .obstruction_detector import ObstructionDetector
from .frame_selector import select_frames, select_frames_normal
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

    def process_single(self, video_filename: str):
        video_path = os.path.join(self.config.video_directory, video_filename)
        if not os.path.isfile(video_path):
            self.progress_callback(0, f"Video not found: {video_path}")
            return
        self.progress_callback(0, f"Processing {video_filename}...")
        try:
            self._process_video(video_path, video_filename)
        except Exception as e:
            self.progress_callback(0, f"Error processing {video_filename}: {e}")
        self.progress_callback(100, f"Done. Processed {video_filename}.")

    def process_selected(self, video_filenames: List[str]):
        total = len(video_filenames)
        for i, vf in enumerate(video_filenames):
            if self.cancel_check():
                self.progress_callback(0, "Processing cancelled.")
                return
            pct = int((i / total) * 100)
            self.progress_callback(pct, f"Processing {vf} ({i + 1}/{total})...")
            video_path = os.path.join(self.config.video_directory, vf)
            try:
                self._process_video(video_path, vf)
            except Exception as e:
                self.progress_callback(pct, f"Error processing {vf}: {e}")
        self.progress_callback(100, f"Done. Processed {total} video(s).")

    def _process_video(self, video_path: str, video_filename: str):
        # Use per-video time frames if configured, otherwise fall back to global
        time_frames = (self.config.video_time_frames.get(video_filename)
                       or self.config.time_frames)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.progress_callback(0, f"Cannot open {video_filename}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.progress_callback(0,
            f"  Video info: {width}x{height}, {fps:.1f} FPS, "
            f"{total_frames} frames, {duration_s:.1f}s")

        if self.config.per_video_start and video_filename in self.config.video_start_marks:
            start_ms = self.config.video_start_marks[video_filename]
        else:
            start_ms = self.config.experiment_start_ms

        self.progress_callback(0,
            f"  Experiment start: {start_ms / 1000:.2f}s")

        if self.config.crop_region:
            cr = self.config.crop_region
            persp_info = ""
            if cr.perspective_x != 0 or cr.perspective_y != 0:
                persp_info = (f", perspective ({cr.perspective_x:.1f}°, "
                              f"{cr.perspective_y:.1f}°)")
            self.progress_callback(0,
                f"  Crop region: ({cr.x}, {cr.y}) {cr.w}x{cr.h}, "
                f"rotation {cr.rotation_angle:.1f} deg{persp_info}")
        else:
            self.progress_callback(0, "  No crop region set, using full frames")

        current_ms = start_ms

        for tf_idx, tf in enumerate(time_frames):
            if self.cancel_check():
                break

            clip_start_s = current_ms / 1000
            clip_end_s = clip_start_s + tf.duration_seconds
            self.progress_callback(0,
                f"  --- Time frame '{tf.name}' ({tf_idx + 1}/"
                f"{len(time_frames)}) ---")
            self.progress_callback(0,
                f"  Clip range: {clip_start_s:.2f}s - {clip_end_s:.2f}s "
                f"(duration: {tf.duration_seconds}s)")

            self.progress_callback(0,
                f"  Streaming frames...")
            cropped_frames, guard_regions, timestamps = self._stream_clip(
                cap, fps, current_ms, tf.duration_seconds,
                skip_first=(tf_idx > 0))

            if self.cancel_check():
                break

            if not cropped_frames:
                self.progress_callback(0,
                    f"  WARNING: No frames extracted for '{tf.name}' "
                    f"- clip may be beyond video end")
                current_ms += tf.duration_seconds * 1000
                continue

            self.progress_callback(0,
                f"  Got {len(cropped_frames)} cropped frames"
                + (f" (size: {cropped_frames[0].shape[1]}x"
                   f"{cropped_frames[0].shape[0]})"
                   if cropped_frames else ""))

            # Save unfiltered frames for debugging
            base = os.path.splitext(video_filename)[0]
            if self.config.save_unfiltered:
                unfiltered_dir = os.path.join(
                    self.config.output_directory, base, tf.name, "_unfiltered")
                os.makedirs(unfiltered_dir, exist_ok=True)
                self.progress_callback(0,
                    f"  Saving {len(cropped_frames)} unfiltered frames "
                    f"to {unfiltered_dir}")
                for ui, uf in enumerate(cropped_frames):
                    if self.cancel_check():
                        break
                    ts_ms = timestamps[ui] if ui < len(timestamps) else 0.0
                    rel_s = (ts_ms - start_ms) / 1000.0
                    upath = os.path.join(
                        unfiltered_dir,
                        f"{base}_{tf.name}_{ui + 1:04d}_{rel_s:.3f}s.png")
                    cv2.imwrite(upath, uf)

            if self.cancel_check():
                break

            # Detect obstructions
            if self.config.obstruction_enabled:
                self.progress_callback(0,
                    f"  Running obstruction detection "
                    f"(sensitivity: {self.config.obstruction_sensitivity:.2f})...")
                detector = ObstructionDetector(
                    sensitivity=self.config.obstruction_sensitivity)
                good_indices = detector.filter_frames(
                    cropped_frames,
                    guard_regions=guard_regions,
                    cancel_check=self.cancel_check)
                if self.cancel_check():
                    break
                rejected = len(cropped_frames) - len(good_indices)
                self.progress_callback(0,
                    f"  Obstruction filter: {len(good_indices)} good, "
                    f"{rejected} rejected")
            else:
                self.progress_callback(0, "  Obstruction detection: disabled")
                good_indices = list(range(len(cropped_frames)))

            # Save filtered/rejected frames for debugging (only when obstruction detection ran)
            if self.config.save_unfiltered and self.config.obstruction_enabled:
                good_set = set(good_indices)
                rejected_indices = [i for i in range(len(cropped_frames))
                                    if i not in good_set]

                if good_indices:
                    filtered_dir = os.path.join(
                        self.config.output_directory, base, tf.name, "_filtered")
                    os.makedirs(filtered_dir, exist_ok=True)
                    self.progress_callback(0,
                        f"  Saving {len(good_indices)} filtered frames "
                        f"to {filtered_dir}")
                    for fi, frame_idx in enumerate(good_indices):
                        if self.cancel_check():
                            break
                        ts_ms = (timestamps[frame_idx]
                                 if frame_idx < len(timestamps) else 0.0)
                        rel_s = (ts_ms - start_ms) / 1000.0
                        fpath = os.path.join(
                            filtered_dir,
                            f"{base}_{tf.name}_{fi + 1:04d}_{rel_s:.3f}s.png")
                        cv2.imwrite(fpath, cropped_frames[frame_idx])

                if rejected_indices:
                    rejected_dir = os.path.join(
                        self.config.output_directory, base, tf.name, "_rejected")
                    os.makedirs(rejected_dir, exist_ok=True)
                    self.progress_callback(0,
                        f"  Saving {len(rejected_indices)} rejected frames "
                        f"to {rejected_dir}")
                    for ri, frame_idx in enumerate(rejected_indices):
                        if self.cancel_check():
                            break
                        ts_ms = (timestamps[frame_idx]
                                 if frame_idx < len(timestamps) else 0.0)
                        rel_s = (ts_ms - start_ms) / 1000.0
                        rpath = os.path.join(
                            rejected_dir,
                            f"{base}_{tf.name}_{ri + 1:04d}_{rel_s:.3f}s.png")
                        cv2.imwrite(rpath, cropped_frames[frame_idx])

            if self.cancel_check():
                break

            # Select frames — uniform or normal-distribution weighted
            if self.config.normal_distribution_mode:
                selected = select_frames_normal(good_indices, tf.num_frames)
                self.progress_callback(0,
                    f"  Selecting {len(selected)}/{tf.num_frames} requested "
                    f"frames (normal distribution, peak at centre of clip)")
            elif len(good_indices) > 0 and tf.num_frames > 0:
                selected = select_frames(good_indices, tf.num_frames)
                interval = len(good_indices) / max(tf.num_frames, 1)
                self.progress_callback(0,
                    f"  Selecting {len(selected)}/{tf.num_frames} requested "
                    f"frames (uniform, interval: every {interval:.1f} good frames)")
            else:
                selected = select_frames(good_indices, tf.num_frames)
                self.progress_callback(0,
                    f"  No frames available to select")

            if self.cancel_check():
                break

            # Save frames
            self._save_frames(cropped_frames, selected, timestamps,
                              video_filename, tf, start_ms)

            current_ms += tf.duration_seconds * 1000

        if self.cancel_check():
            self.progress_callback(0, f"  Cancelled processing {video_filename}")
        else:
            self.progress_callback(0, f"  Finished {video_filename}")
        cap.release()

    def _stream_clip(self, cap: cv2.VideoCapture, fps: float,
                     start_ms: float, duration_s: float,
                     skip_first: bool = False):
        """Read frames one at a time, rotate+crop immediately, only keep
        the small cropped result. Never holds full-resolution frames in memory.

        Returns (cropped_frames, guard_regions, timestamps).
        guard_regions is a list of border-ring images (same length as
        cropped_frames) used for obstruction detection, or None when
        no crop region is set.
        """
        crop = self.config.crop_region
        cropped_frames = []
        guard_regions = [] if crop else None
        timestamps = []

        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        end_ms = start_ms + duration_s * 1000

        bbox = (crop.x, crop.y, crop.w, crop.h) if crop else None

        first = True
        frame_count = 0
        expected = int(duration_s * fps)
        log_interval = max(1, expected // 5)

        while True:
            if self.cancel_check():
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms >= end_ms:
                break
            ret, raw_frame = cap.read()
            if not ret:
                break
            # OpenCV seek snaps to the nearest keyframe which can be before
            # start_ms — discard those frames silently.
            if pos_ms < start_ms:
                raw_frame = None
                continue
            if first and skip_first:
                first = False
                raw_frame = None
                continue
            first = False

            # Apply perspective then rotation (on single frame, discard raw)
            frame = raw_frame
            if crop and (crop.perspective_x != 0 or crop.perspective_y != 0):
                frame = self._perspective_warp(
                    frame, crop.perspective_x, crop.perspective_y)
            if crop and crop.rotation_angle != 0:
                frame = self._rotate_frame(frame, crop.rotation_angle)
            raw_frame = None

            if crop:
                x, y, w, h = bbox
                fh, fw = frame.shape[:2]
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                w = min(w, fw - x)
                h = min(h, fh - y)

                # Extract guard border before cropping
                guard = ObstructionDetector.extract_guard_region(
                    frame, x, y, w, h,
                    margin=40)
                guard_regions.append(guard)

                crop_img = frame[y:y + h, x:x + w].copy()
                frame = None
                cropped_frames.append(crop_img)
            else:
                cropped_frames.append(frame)

            timestamps.append(pos_ms)
            frame_count += 1

            if frame_count % log_interval == 0:
                self.progress_callback(0,
                    f"    {frame_count}/{expected} frames processed...")

        return cropped_frames, guard_regions, timestamps

    @staticmethod
    def _perspective_warp(frame: np.ndarray,
                          angle_x: float, angle_y: float) -> np.ndarray:
        h, w = frame.shape[:2]
        dx = np.tan(np.radians(angle_x)) * w * 0.3
        dy = np.tan(np.radians(angle_y)) * h * 0.3
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [0 + dx, 0 + dy],
            [w - dx, 0 - dy],
            [w + dx, h + dy],
            [0 - dx, h - dy],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (w, h))

    @staticmethod
    def _rotate_frame(frame: np.ndarray, angle: float) -> np.ndarray:
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
                     timestamps: List[float], video_filename: str, tf: TimeFrame,
                     start_ms: float = 0.0):
        video_base = os.path.splitext(video_filename)[0]
        out_dir = os.path.join(
            self.config.output_directory, video_base, tf.name)
        os.makedirs(out_dir, exist_ok=True)
        if self.config.save_unfiltered and self.config.filter_manually:
            os.makedirs(os.path.join(out_dir, "_manually-filtered"), exist_ok=True)
        self.progress_callback(0, f"  Saving to: {out_dir}")

        for save_idx, frame_idx in enumerate(selected_indices):
            if self.cancel_check():
                self.progress_callback(0, "  Save interrupted by cancel")
                return
            if frame_idx < len(frames):
                ts = timestamps[frame_idx] if frame_idx < len(timestamps) else 0.0
                rel_s = (ts - start_ms) / 1000.0
                filename = generate_filename(
                    tf.naming_scheme, video_filename, tf.name, save_idx, ts,
                    rel_time_s=rel_s)
                filepath = os.path.join(out_dir, filename)
                cv2.imwrite(filepath, frames[frame_idx])
                h, w = frames[frame_idx].shape[:2]
                self.progress_callback(0,
                    f"    [{save_idx + 1}/{len(selected_indices)}] "
                    f"{filename} ({w}x{h}, t={rel_s:.3f}s from start)")

        self.progress_callback(0,
            f"  Saved {len(selected_indices)} frames for '{tf.name}'")
