import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QStyle, QSizePolicy, QFileDialog, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import os


class VideoPlayer(QWidget):
    experiment_marked = pyqtSignal(float)  # emits timestamp in ms
    frame_for_crop = pyqtSignal(np.ndarray)  # emits the frame at marked position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_num = 0
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.marked_ms = None
        self.video_dir = ""
        self.video_files = []
        self.current_video_path = ""

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Video:"))
        self.video_combo = QComboBox()
        self.video_combo.currentIndexChanged.connect(self._on_video_selected)
        selector_layout.addWidget(self.video_combo, 1)
        layout.addLayout(selector_layout)

        # Display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 360)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.display_label, 1)

        # Seek bar
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(self._seek)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.seek_slider)

        # Time label
        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)

        # Controls
        ctrl_layout = QHBoxLayout()

        self.btn_step_back = QPushButton("⏮ -1s")
        self.btn_step_back.clicked.connect(lambda: self._step(-1.0))
        ctrl_layout.addWidget(self.btn_step_back)

        self.btn_frame_back = QPushButton("◀ Frame")
        self.btn_frame_back.clicked.connect(lambda: self._step_frames(-1))
        ctrl_layout.addWidget(self.btn_frame_back)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        self.btn_frame_fwd = QPushButton("Frame ▶")
        self.btn_frame_fwd.clicked.connect(lambda: self._step_frames(1))
        ctrl_layout.addWidget(self.btn_frame_fwd)

        self.btn_step_fwd = QPushButton("+1s ⏭")
        self.btn_step_fwd.clicked.connect(lambda: self._step(1.0))
        ctrl_layout.addWidget(self.btn_step_fwd)

        layout.addLayout(ctrl_layout)

        # Mark button
        mark_layout = QHBoxLayout()
        self.btn_mark = QPushButton("🔴 Mark Experiment Start Here")
        self.btn_mark.setStyleSheet(
            "QPushButton { background-color: #d32f2f; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #b71c1c; }")
        self.btn_mark.clicked.connect(self._mark_start)
        mark_layout.addWidget(self.btn_mark)

        self.mark_label = QLabel("No mark set")
        self.mark_label.setStyleSheet("color: gray;")
        mark_layout.addWidget(self.mark_label)

        layout.addLayout(mark_layout)

    def set_video_directory(self, directory: str):
        self.video_dir = directory
        extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
        self.video_files = [f for f in sorted(os.listdir(directory))
                            if f.lower().endswith(extensions)]
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for vf in self.video_files:
            self.video_combo.addItem(vf)
        self.video_combo.blockSignals(False)
        if self.video_files:
            self.video_combo.setCurrentIndex(0)
            self._load_video(os.path.join(directory, self.video_files[0]))

    def _on_video_selected(self, index):
        if 0 <= index < len(self.video_files):
            path = os.path.join(self.video_dir, self.video_files[index])
            self._load_video(path)

    def _load_video(self, path: str):
        self._stop()
        if self.cap:
            self.cap.release()

        self.current_video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.display_label.setText(f"Cannot open video")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.seek_slider.setRange(0, max(0, self.total_frames - 1))
        self.current_frame_num = 0
        self._show_current_frame()

    def _show_current_frame(self):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
            self._update_time_label()
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.current_frame_num)
            self.seek_slider.blockSignals(False)

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        label_size = self.display_label.size()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled)

    def _update_time_label(self):
        if self.cap is None:
            return
        current_ms = (self.current_frame_num / self.fps) * 1000
        total_ms = (self.total_frames / self.fps) * 1000
        self.time_label.setText(
            f"{self._format_time(current_ms)} / {self._format_time(total_ms)}")

    def _format_time(self, ms: float) -> str:
        total_s = ms / 1000.0
        minutes = int(total_s // 60)
        seconds = total_s % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    def _toggle_play(self):
        if self.is_playing:
            self._stop()
        else:
            self._play()

    def _play(self):
        if self.cap is None:
            return
        self.is_playing = True
        self.btn_play.setText("⏸ Pause")
        interval = max(1, int(1000 / self.fps))
        self.timer.start(interval)

    def _stop(self):
        self.is_playing = False
        self.btn_play.setText("▶ Play")
        self.timer.stop()

    def _next_frame(self):
        if self.cap is None:
            self._stop()
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._display_frame(frame)
            self._update_time_label()
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.current_frame_num)
            self.seek_slider.blockSignals(False)
        else:
            self._stop()

    def _seek(self, frame_num: int):
        self.current_frame_num = frame_num
        self._show_current_frame()

    def _on_slider_pressed(self):
        if self.is_playing:
            self.timer.stop()

    def _on_slider_released(self):
        if self.is_playing:
            interval = max(1, int(1000 / self.fps))
            self.timer.start(interval)

    def _step(self, seconds: float):
        if self.cap is None:
            return
        delta_frames = int(seconds * self.fps)
        self.current_frame_num = max(0, min(
            self.total_frames - 1, self.current_frame_num + delta_frames))
        self._show_current_frame()

    def _step_frames(self, n: int):
        if self.cap is None:
            return
        self.current_frame_num = max(0, min(
            self.total_frames - 1, self.current_frame_num + n))
        self._show_current_frame()

    def _mark_start(self):
        if self.cap is None:
            return
        self.marked_ms = (self.current_frame_num / self.fps) * 1000
        self.mark_label.setText(
            f"Marked at {self._format_time(self.marked_ms)}")
        self.mark_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.experiment_marked.emit(self.marked_ms)

        # Send current frame for crop setup
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.frame_for_crop.emit(frame)

    def get_marked_ms(self) -> float:
        return self.marked_ms if self.marked_ms is not None else 0.0

    def get_current_video_name(self) -> str:
        if self.current_video_path:
            return os.path.basename(self.current_video_path)
        return ""

    def cleanup(self):
        self._stop()
        if self.cap:
            self.cap.release()
            self.cap = None
