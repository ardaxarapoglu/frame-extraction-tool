from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSlider, QGroupBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
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

        # Per-video marking option
        self.per_video_check = QCheckBox(
            "Mark experiment start separately for each video")
        layout.addWidget(self.per_video_check)

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

    def is_per_video(self) -> bool:
        return self.per_video_check.isChecked()
