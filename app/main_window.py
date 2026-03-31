import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox, QTabWidget, QDialog
)
from PyQt5.QtCore import Qt

from .widgets.settings_panel import SettingsPanel
from .widgets.video_player import VideoPlayer
from .widgets.crop_rotate_widget import CropRotateWidget
from .widgets.progress_panel import ProgressPanel
from .workers.processing_worker import ProcessingWorker
from .dialogs.obstruction_test_dialog import ObstructionTestDialog
from .core.models import ProjectConfig, CropRegion


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Froth Flotation Frame Extractor")
        self.setMinimumSize(1200, 800)
        self.config = ProjectConfig()
        self.worker = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Main splitter: settings | video+crop
        splitter = QSplitter(Qt.Horizontal)

        # Left: Settings panel
        self.settings_panel = SettingsPanel()
        self.settings_panel.setMinimumWidth(320)
        self.settings_panel.setMaximumWidth(450)
        splitter.addWidget(self.settings_panel)

        # Right: Video player + crop area in tabs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1: Video Player
        self.video_player = VideoPlayer()
        self.tabs.addTab(self.video_player, "Video Player")

        # Tab 2: Crop & Rotate
        self.crop_widget = CropRotateWidget()
        self.tabs.addTab(self.crop_widget, "Crop && Rotate")

        right_layout.addWidget(self.tabs, 1)

        # Progress panel at bottom
        self.progress_panel = ProgressPanel()
        right_layout.addWidget(self.progress_panel)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _connect_signals(self):
        # Video dir browsing
        self.settings_panel.video_dir_changed.connect(
            self._on_video_dir_changed)

        # Mark experiment start
        self.video_player.experiment_marked.connect(
            self._on_experiment_marked)
        self.video_player.frame_for_crop.connect(
            self._on_frame_for_crop)

        # Crop applied
        self.crop_widget.crop_applied.connect(self._on_crop_applied)

        # Process
        self.settings_panel.process_requested.connect(self._start_processing)
        self.settings_panel.process_single_requested.connect(
            self._start_processing_single)

        # Test obstruction
        self.settings_panel.test_obstruction_requested.connect(
            self._test_obstruction)

        # Cancel
        self.progress_panel.cancel_requested.connect(self._cancel_processing)

    def _on_video_dir_changed(self, directory: str):
        self.config.video_directory = directory
        self.video_player.set_video_directory(directory)

    def _on_experiment_marked(self, ms: float):
        if self.settings_panel.is_per_video():
            video_name = self.video_player.get_current_video_name()
            self.config.video_start_marks[video_name] = ms
        else:
            self.config.experiment_start_ms = ms

    def _on_frame_for_crop(self, frame: np.ndarray):
        self.crop_widget.set_frame(frame)
        self.tabs.setCurrentIndex(1)  # Switch to crop tab

    def _on_crop_applied(self, region: CropRegion):
        self.config.crop_region = region

    def _test_obstruction(self):
        import cv2
        # Grab ~3 seconds of frames from current video position
        cap = self.video_player.cap
        if cap is None or not cap.isOpened():
            QMessageBox.warning(self, "No Video",
                                "Open a video first.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Save and restore position
        saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Sample ~3 seconds, every 3rd frame to keep it quick
        sample_count = int(fps * 3)
        frames = []
        for _ in range(sample_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        # Restore position
        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

        if not frames:
            QMessageBox.warning(self, "No Frames",
                                "Could not read frames from current position.")
            return

        # Apply crop if set
        crop = self.config.crop_region
        if crop:
            cropped = []
            for frame in frames:
                if crop.rotation_angle != 0:
                    h, w = frame.shape[:2]
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, crop.rotation_angle, 1.0)
                    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
                    new_w = int(h * sin_a + w * cos_a)
                    new_h = int(h * cos_a + w * sin_a)
                    M[0, 2] += (new_w - w) / 2
                    M[1, 2] += (new_h - h) / 2
                    frame = cv2.warpAffine(frame, M, (new_w, new_h))
                x, y, cw, ch = crop.x, crop.y, crop.w, crop.h
                fh, fw = frame.shape[:2]
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                cw = min(cw, fw - x)
                ch = min(ch, fh - y)
                cropped.append(frame[y:y + ch, x:x + cw])
            frames = cropped

        # Take every 3rd frame to keep the dialog snappy
        frames = frames[::3]
        if not frames:
            QMessageBox.warning(self, "No Frames",
                                "No frames after cropping.")
            return

        sensitivity = self.settings_panel.get_sensitivity()
        dlg = ObstructionTestDialog(frames, sensitivity, self)
        if dlg.exec_() == QDialog.Accepted:
            new_sens = dlg.get_sensitivity()
            self.settings_panel.sensitivity_slider.setValue(
                int(new_sens * 100))

    def _start_processing_single(self):
        video_name = self.video_player.get_current_video_name()
        if not video_name:
            QMessageBox.warning(self, "No Video",
                                "No video is currently selected.")
            return
        self._start_processing(single_video=video_name)

    def _start_processing(self, single_video=None):
        # Validate
        if not self.config.video_directory:
            QMessageBox.warning(self, "Missing Setting",
                                "Please select a video directory.")
            return

        output_dir = self.settings_panel.get_output_directory()
        if not output_dir:
            QMessageBox.warning(self, "Missing Setting",
                                "Please select an output directory.")
            return

        time_frames = self.settings_panel.time_frame_editor.get_time_frames()
        if not time_frames:
            QMessageBox.warning(self, "Missing Setting",
                                "Please add at least one time frame.")
            return

        if self.config.crop_region is None:
            reply = QMessageBox.question(
                self, "No Crop Region",
                "No crop region is set. Frames will not be cropped.\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Build config
        self.config.output_directory = output_dir
        self.config.time_frames = time_frames
        self.config.obstruction_enabled = \
            self.settings_panel.is_obstruction_enabled()
        self.config.obstruction_sensitivity = \
            self.settings_panel.get_sensitivity()
        self.config.per_video_start = self.settings_panel.is_per_video()

        # Start worker
        self.progress_panel.clear_log()
        self.progress_panel.set_processing(True)
        self.settings_panel.btn_process.setEnabled(False)
        self.settings_panel.btn_process_single.setEnabled(False)

        self.worker = ProcessingWorker(self.config,
                                       single_video=single_video)
        self.worker.progress_updated.connect(
            self.progress_panel.set_progress)
        self.worker.finished_processing.connect(
            self._on_processing_finished)
        self.worker.error_occurred.connect(
            self._on_processing_error)
        self.worker.start()

    def _cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.progress_panel.set_progress(0, "Cancelling...")

    def _on_processing_finished(self):
        self.progress_panel.set_processing(False)
        self.settings_panel.btn_process.setEnabled(True)
        self.settings_panel.btn_process_single.setEnabled(True)
        if self.worker and self.worker._cancelled:
            self.progress_panel.set_progress(0, "Cancelled.")
        else:
            self.progress_panel.set_progress(0, "Processing complete!")

    def _on_processing_error(self, error: str):
        QMessageBox.critical(self, "Processing Error", error)

    def closeEvent(self, event):
        self.video_player.cleanup()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()
