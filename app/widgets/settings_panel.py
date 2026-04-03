import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSlider, QGroupBox, QCheckBox,
    QComboBox, QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QSettings
from .time_frame_editor import TimeFrameEditor


class SettingsPanel(QWidget):
    video_dir_changed = pyqtSignal(str)
    process_requested = pyqtSignal()
    process_single_requested = pyqtSignal()
    test_obstruction_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Video directory
        vid_group = QGroupBox("Video Directory")
        vid_layout = QHBoxLayout(vid_group)
        self.video_dir_edit = QLineEdit()
        self.video_dir_edit.setPlaceholderText("Select video directory...")
        self.video_dir_edit.setReadOnly(True)
        vid_layout.addWidget(self.video_dir_edit)
        btn_vid = QPushButton("Browse...")
        btn_vid.clicked.connect(self._browse_video_dir)
        vid_layout.addWidget(btn_vid)
        layout.addWidget(vid_group)

        # Output directory
        out_group = QGroupBox("Output Directory")
        out_layout = QHBoxLayout(out_group)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_dir_edit.setReadOnly(True)
        out_layout.addWidget(self.output_dir_edit)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(btn_out)
        layout.addWidget(out_group)

        # Time frames
        tf_group = QGroupBox("Time Frame Settings")
        tf_layout = QVBoxLayout(tf_group)
        self.time_frame_editor = TimeFrameEditor()
        tf_layout.addWidget(self.time_frame_editor)
        layout.addWidget(tf_group)

        # Obstruction detection
        obs_group = QGroupBox("Obstruction Detection")
        obs_layout = QVBoxLayout(obs_group)

        self.obstruction_enabled_check = QCheckBox("Enable obstruction detection")
        self.obstruction_enabled_check.setChecked(True)
        self.obstruction_enabled_check.toggled.connect(self._on_obstruction_toggled)
        obs_layout.addWidget(self.obstruction_enabled_check)

        self.obs_controls_widget = QWidget()
        obs_controls_layout = QVBoxLayout(self.obs_controls_widget)
        obs_controls_layout.setContentsMargins(0, 0, 0, 0)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 80)
        self.sensitivity_slider.setValue(35)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(10)
        slider_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label = QLabel("0.35")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v / 100:.2f}"))
        slider_layout.addWidget(self.sensitivity_label)
        obs_controls_layout.addLayout(slider_layout)

        self.btn_test_obstruction = QPushButton("Test Obstruction Detection...")
        self.btn_test_obstruction.setStyleSheet(
            "QPushButton { background-color: #f57c00; color: white; "
            "font-weight: bold; padding: 6px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e65100; }")
        self.btn_test_obstruction.clicked.connect(
            self.test_obstruction_requested.emit)
        obs_controls_layout.addWidget(self.btn_test_obstruction)

        obs_layout.addWidget(self.obs_controls_widget)
        layout.addWidget(obs_group)

        # Tracking toggle
        self.tracking_enabled_check = QCheckBox(
            "Track crop region for camera movement")
        self.tracking_enabled_check.setChecked(True)
        layout.addWidget(self.tracking_enabled_check)

        # Per-video marking option
        self.per_video_check = QCheckBox(
            "Mark experiment start separately for each video")
        layout.addWidget(self.per_video_check)

        # Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)

        preset_select_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setPlaceholderText("Select a preset...")
        preset_select_layout.addWidget(self.preset_combo, 1)

        self.btn_load_preset = QPushButton("Load")
        self.btn_load_preset.clicked.connect(self._load_preset)
        preset_select_layout.addWidget(self.btn_load_preset)
        preset_layout.addLayout(preset_select_layout)

        preset_btn_layout = QHBoxLayout()
        self.btn_save_preset = QPushButton("Save Current Settings")
        self.btn_save_preset.clicked.connect(self._save_preset)
        preset_btn_layout.addWidget(self.btn_save_preset)

        self.btn_delete_preset = QPushButton("Delete")
        self.btn_delete_preset.clicked.connect(self._delete_preset)
        preset_btn_layout.addWidget(self.btn_delete_preset)
        preset_layout.addLayout(preset_btn_layout)

        layout.addWidget(preset_group)
        self._refresh_preset_list()

        # Process buttons
        process_layout = QHBoxLayout()

        self.btn_process_single = QPushButton("▶ Process Selected Video")
        self.btn_process_single.setStyleSheet(
            "QPushButton { background-color: #388e3c; color: white; "
            "font-weight: bold; padding: 10px; font-size: 13px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background-color: #2e7d32; }"
            "QPushButton:disabled { background-color: #90a4ae; }")
        self.btn_process_single.clicked.connect(
            self.process_single_requested.emit)
        process_layout.addWidget(self.btn_process_single)

        self.btn_process = QPushButton("▶ Process All Videos")
        self.btn_process.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; "
            "font-weight: bold; padding: 10px; font-size: 13px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:disabled { background-color: #90a4ae; }")
        self.btn_process.clicked.connect(self.process_requested.emit)
        process_layout.addWidget(self.btn_process)

        layout.addLayout(process_layout)

        # Debug: save unfiltered + filtered frames
        self.save_unfiltered_check = QCheckBox(
            "Save all frames before obstruction filtering (debug)")
        layout.addWidget(self.save_unfiltered_check)

        layout.addStretch()

    def _browse_video_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Video Directory")
        if d:
            self.video_dir_edit.setText(d)
            self.video_dir_changed.emit(d)

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.output_dir_edit.setText(d)

    def get_video_directory(self) -> str:
        return self.video_dir_edit.text()

    def get_output_directory(self) -> str:
        return self.output_dir_edit.text()

    def get_sensitivity(self) -> float:
        return self.sensitivity_slider.value() / 100.0

    def is_obstruction_enabled(self) -> bool:
        return self.obstruction_enabled_check.isChecked()

    def _on_obstruction_toggled(self, enabled: bool):
        self.obs_controls_widget.setEnabled(enabled)

    def is_tracking_enabled(self) -> bool:
        return self.tracking_enabled_check.isChecked()

    def is_save_unfiltered(self) -> bool:
        return self.save_unfiltered_check.isChecked()

    def is_per_video(self) -> bool:
        return self.per_video_check.isChecked()

    def _get_settings(self):
        return QSettings("FrothExtractor", "FrothFrameExtractor")

    def _refresh_preset_list(self):
        self.preset_combo.clear()
        settings = self._get_settings()
        settings.beginGroup("presets")
        names = settings.childKeys()
        settings.endGroup()
        for name in sorted(names):
            self.preset_combo.addItem(name)

    def _save_preset(self):
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        # Gather current settings
        time_frames = []
        for tf in self.time_frame_editor.get_time_frames():
            time_frames.append({
                "name": tf.name,
                "duration_seconds": tf.duration_seconds,
                "num_frames": tf.num_frames,
                "naming_scheme": tf.naming_scheme,
            })

        preset = {
            "time_frames": time_frames,
            "obstruction_enabled": self.obstruction_enabled_check.isChecked(),
            "obstruction_sensitivity": self.sensitivity_slider.value(),
            "tracking_enabled": self.tracking_enabled_check.isChecked(),
            "per_video_start": self.per_video_check.isChecked(),
        }

        settings = self._get_settings()
        settings.setValue(f"presets/{name}", json.dumps(preset))
        self._refresh_preset_list()

        # Select the saved preset
        idx = self.preset_combo.findText(name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    def _load_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            return

        settings = self._get_settings()
        data = settings.value(f"presets/{name}")
        if not data:
            return

        try:
            preset = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            QMessageBox.warning(self, "Error", "Could not load preset data.")
            return

        # Apply time frames
        from ..core.models import TimeFrame
        tfs = []
        for tf_data in preset.get("time_frames", []):
            tfs.append(TimeFrame(
                name=tf_data.get("name", "Phase1"),
                duration_seconds=tf_data.get("duration_seconds", 30.0),
                num_frames=tf_data.get("num_frames", 5),
                naming_scheme=tf_data.get("naming_scheme",
                                          "{video}_{name}_{index:03d}"),
            ))
        self.time_frame_editor.set_time_frames(tfs)

        # Apply obstruction settings
        self.obstruction_enabled_check.setChecked(
            preset.get("obstruction_enabled", True))
        self.sensitivity_slider.setValue(
            preset.get("obstruction_sensitivity", 35))

        # Apply tracking
        self.tracking_enabled_check.setChecked(
            preset.get("tracking_enabled", True))

        # Apply per-video start
        self.per_video_check.setChecked(
            preset.get("per_video_start", False))

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Delete preset '{name}'?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            settings = self._get_settings()
            settings.remove(f"presets/{name}")
            self._refresh_preset_list()
