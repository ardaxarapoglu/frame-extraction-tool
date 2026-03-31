import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt

from .widgets.settings_panel import SettingsPanel
from .widgets.video_player import VideoPlayer
from .widgets.crop_rotate_widget import CropRotateWidget
from .widgets.progress_panel import ProgressPanel
from .workers.processing_worker import ProcessingWorker
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

    def _start_processing(self):
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
        self.config.obstruction_sensitivity = \
            self.settings_panel.get_sensitivity()
        self.config.per_video_start = self.settings_panel.is_per_video()

        # Start worker
        self.progress_panel.clear_log()
        self.progress_panel.set_processing(True)
        self.settings_panel.btn_process.setEnabled(False)

        self.worker = ProcessingWorker(self.config)
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

    def _on_processing_finished(self):
        self.progress_panel.set_processing(False)
        self.settings_panel.btn_process.setEnabled(True)
        self.progress_panel.set_progress(100, "Processing complete!")

    def _on_processing_error(self, error: str):
        QMessageBox.critical(self, "Processing Error", error)

    def closeEvent(self, event):
        self.video_player.cleanup()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()
