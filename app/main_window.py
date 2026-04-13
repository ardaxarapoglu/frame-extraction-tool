import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from .core.models import ProjectConfig, CropRegion
from .core.video_dir_config import VideoDirectoryConfig
from .dialogs.select_videos_dialog import SelectVideosDialog
from .dialogs.video_timeframes_dialog import VideoTimeFramesDialog
from .widgets.crop_rotate_widget import CropRotateWidget
from .widgets.debug_panel import DebugPanel
from .widgets.progress_panel import ProgressPanel
from .widgets.settings_panel import SettingsPanel
from .widgets.video_player import VideoPlayer
from .workers.processing_worker import ProcessingWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Froth Flotation Frame Extractor")
        self.setMinimumSize(1200, 800)
        self.config = ProjectConfig()
        self._vdc = VideoDirectoryConfig()
        self.worker = None
        self._loading = False  # suppresses auto-save during config restore

        # Debounce timer: saves config 1.5 s after last settings change
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1500)
        self._save_timer.timeout.connect(self._save_config)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        self.settings_panel = SettingsPanel()
        self.settings_panel.setMinimumWidth(320)
        self.settings_panel.setMaximumWidth(450)
        splitter.addWidget(self.settings_panel)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        self.video_player = VideoPlayer()
        self.tabs.addTab(self.video_player, "Video Player")

        self.crop_widget = CropRotateWidget()
        self.tabs.addTab(self.crop_widget, "Crop && Rotate")

        self.progress_panel = ProgressPanel()
        self.tabs.addTab(self.progress_panel, "Processing Log")

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
        # Video dir
        self.settings_panel.video_dir_changed.connect(self._on_video_dir_changed)

        # Video player events
        self.video_player.experiment_marked.connect(self._on_experiment_marked)
        self.video_player.frame_for_crop.connect(self._on_frame_for_crop)
        self.video_player.video_selected.connect(self._on_video_selected)
        self.video_player.edit_video_timeframes_requested.connect(
            self._on_edit_video_timeframes)

        # Crop applied
        self.crop_widget.crop_applied.connect(self._on_crop_applied)

        # Debug panel output dir
        self.settings_panel.output_dir_edit.textChanged.connect(
            self.debug_panel.set_output_dir)

        # Process buttons
        self.settings_panel.process_multiple_requested.connect(
            self._on_process_multiple)
        self.settings_panel.process_single_requested.connect(
            self._start_processing_single)

        # Cancel
        self.progress_panel.cancel_requested.connect(self._cancel_processing)

        # Auto-save on settings changes (debounced)
        self.settings_panel.output_dir_edit.textChanged.connect(
            self._schedule_save)
        self.settings_panel.time_frame_editor.changed.connect(
            self._schedule_save)
        self.settings_panel.obstruction_enabled_check.toggled.connect(
            self._schedule_save)
        self.settings_panel.sensitivity_slider.valueChanged.connect(
            self._schedule_save)
        self.settings_panel.tracking_enabled_check.toggled.connect(
            self._schedule_save)
        self.settings_panel.save_unfiltered_check.toggled.connect(
            self._schedule_save)
        self.settings_panel.per_video_check.toggled.connect(
            self._schedule_save)

    # ------------------------------------------------------------------ #
    # Config persistence
    # ------------------------------------------------------------------ #

    def _schedule_save(self):
        if not self._loading and self._vdc.loaded:
            self._save_timer.start()

    def _save_config(self):
        if not self._vdc.loaded:
            return
        self._vdc.set("output_directory",
                      self.settings_panel.get_output_directory())
        self._vdc.set_global_time_frames(
            self.settings_panel.time_frame_editor.get_time_frames())
        self._vdc.set("obstruction_enabled",
                      self.settings_panel.is_obstruction_enabled())
        self._vdc.set("obstruction_sensitivity",
                      self.settings_panel.get_sensitivity())
        self._vdc.set("tracking_enabled",
                      self.settings_panel.is_tracking_enabled())
        self._vdc.set("save_unfiltered",
                      self.settings_panel.is_save_unfiltered())
        self._vdc.set("per_video_start",
                      self.settings_panel.is_per_video())
        self._vdc.save()

    # ------------------------------------------------------------------ #
    # Video directory
    # ------------------------------------------------------------------ #

    def _on_video_dir_changed(self, directory: str):
        self.config.video_directory = directory

        # Load (or create fresh) the per-directory config
        self._vdc.load(directory)

        # Restore UI from saved settings (suppress auto-save during restore)
        self._loading = True
        try:
            self.settings_panel.apply_config(self._vdc)
        finally:
            self._loading = False

        # Restore all per-video start marks and custom time frames into config
        self.config.video_start_marks = self._vdc.get_all_video_start_marks()
        self.config.video_time_frames = self._vdc.get_all_custom_time_frames()

        # Update the video player list
        self.video_player.set_video_directory(directory)

    # ------------------------------------------------------------------ #
    # Video player events
    # ------------------------------------------------------------------ #

    def _on_video_selected(self, video_name: str):
        """Called whenever the player switches to a different video."""
        # Restore saved start mark
        saved_ms = self._vdc.get_video_start_ms(video_name)
        if saved_ms is not None:
            self.video_player.set_mark_display(saved_ms)

        # Show custom TF status
        custom_tfs = self._vdc.get_video_time_frames(video_name)
        self.video_player.set_timeframes_status(
            bool(custom_tfs), len(custom_tfs) if custom_tfs else 0)

        # Restore saved crop region into config and crop widget
        saved_crop = self._vdc.get_video_crop_region(video_name)
        self.config.crop_region = saved_crop  # None is fine — means no crop
        if saved_crop:
            frame = self.video_player.get_current_frame()
            if frame is not None:
                self.crop_widget.set_frame(frame)
                self.crop_widget.load_crop_region(saved_crop)

    def _on_experiment_marked(self, ms: float):
        video_name = self.video_player.get_current_video_name()
        if self.settings_panel.is_per_video() and video_name:
            self.config.video_start_marks[video_name] = ms
            self._vdc.set_video_start_ms(video_name, ms)
        else:
            self.config.experiment_start_ms = ms
            self._vdc.set("experiment_start_ms", ms)
        self._vdc.save()

    def _on_frame_for_crop(self, frame: np.ndarray):
        self.crop_widget.set_frame(frame)
        # Re-apply any saved crop so the user sees it immediately
        video_name = self.video_player.get_current_video_name()
        if video_name:
            saved_crop = self._vdc.get_video_crop_region(video_name)
            if saved_crop:
                self.crop_widget.load_crop_region(saved_crop)
        self.tabs.setCurrentIndex(1)

    def _on_crop_applied(self, region: CropRegion):
        self.config.crop_region = region
        # Save per-video crop to config file
        video_name = self.video_player.get_current_video_name()
        if video_name and self._vdc.loaded:
            self._vdc.set_video_crop_region(video_name, region)
            self._vdc.save()

    # ------------------------------------------------------------------ #
    # Per-video time frames
    # ------------------------------------------------------------------ #

    def _on_edit_video_timeframes(self, video_name: str):
        custom_tfs = self._vdc.get_video_time_frames(video_name)
        global_tfs = self.settings_panel.time_frame_editor.get_time_frames()

        dlg = VideoTimeFramesDialog(video_name, custom_tfs, global_tfs, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        result = dlg.get_result()  # None = use global, List = custom
        self._vdc.set_video_time_frames(video_name, result)
        self._vdc.save()

        # Keep runtime config in sync
        if result:
            self.config.video_time_frames[video_name] = result
        else:
            self.config.video_time_frames.pop(video_name, None)

        self.video_player.set_timeframes_status(
            bool(result), len(result) if result else 0)

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #

    def _on_process_multiple(self):
        video_files = self.video_player.video_files
        if not video_files:
            QMessageBox.warning(self, "No Videos",
                                "No videos found in the selected directory.")
            return
        dlg = SelectVideosDialog(video_files, self)
        if dlg.exec_() == QDialog.Accepted:
            selected = dlg.get_selected()
            if selected:
                self._start_processing(selected_videos=selected)

    def _start_processing_single(self):
        video_name = self.video_player.get_current_video_name()
        if not video_name:
            QMessageBox.warning(self, "No Video",
                                "No video is currently selected.")
            return
        self._start_processing(single_video=video_name)

    def _start_processing(self, single_video=None, selected_videos=None):
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

        self.config.output_directory = output_dir
        self.config.time_frames = time_frames
        self.config.obstruction_enabled = \
            self.settings_panel.is_obstruction_enabled()
        self.config.obstruction_sensitivity = \
            self.settings_panel.get_sensitivity()
        self.config.tracking_enabled = self.settings_panel.is_tracking_enabled()
        self.config.save_unfiltered = self.settings_panel.is_save_unfiltered()
        self.config.per_video_start = self.settings_panel.is_per_video()

        self.progress_panel.clear_log()
        self.progress_panel.set_processing(True)
        self.tabs.setCurrentIndex(2)
        self.settings_panel.btn_process.setEnabled(False)
        self.settings_panel.btn_process_single.setEnabled(False)

        self.worker = ProcessingWorker(
            self.config,
            single_video=single_video,
            selected_videos=selected_videos,
        )
        self.worker.progress_updated.connect(self.progress_panel.set_progress)
        self.worker.finished_processing.connect(self._on_processing_finished)
        self.worker.error_occurred.connect(self._on_processing_error)
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
        self._save_config()  # flush any pending save
        self.video_player.cleanup()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()
