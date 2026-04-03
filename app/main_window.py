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
from .widgets.debug_panel import DebugPanel
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

        # Tab 3: Processing Log
        self.progress_panel = ProgressPanel()
        self.tabs.addTab(self.progress_panel, "Processing Log")

        # Tab 4: Debug (manual filtering → final selection)
        self.debug_panel = DebugPanel(
            get_time_frames_fn=lambda: (
                self.settings_panel.time_frame_editor.get_time_frames()))
        self.tabs.addTab(self.debug_panel, "Debug")

        right_layout.addWidget(self.tabs, 1)

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

        # Keep debug panel output dir in sync
        self.settings_panel.output_dir_edit.textChanged.connect(
            self.debug_panel.set_output_dir)

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
        cap = self.video_player.cap
        if cap is None or not cap.isOpened():
            QMessageBox.warning(self, "No Video",
                                "Open a video first.")
            return

        sensitivity = self.settings_panel.get_sensitivity()
        dlg = ObstructionTestDialog(
            cap, self.config.crop_region, sensitivity, self)
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
        self.config.tracking_enabled = self.settings_panel.is_tracking_enabled()
        self.config.save_unfiltered = self.settings_panel.is_save_unfiltered()
        self.config.per_video_start = self.settings_panel.is_per_video()

        # Start worker
        self.progress_panel.clear_log()
        self.progress_panel.set_processing(True)
        self.tabs.setCurrentIndex(2)  # Switch to Processing Log tab
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
