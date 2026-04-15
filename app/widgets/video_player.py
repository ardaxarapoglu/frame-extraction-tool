import cv2
import numpy as np
import os
import subprocess
import tempfile
import threading
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSizePolicy, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

try:
    import os as _os
    _os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    pygame.mixer.init()
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False


class VideoPlayer(QWidget):
    experiment_marked = pyqtSignal(float)          # emits timestamp in ms
    frame_for_crop = pyqtSignal(np.ndarray)        # emits the frame at marked position
    video_selected = pyqtSignal(str)               # emits video filename after load
    edit_video_timeframes_requested = pyqtSignal(str)  # emits video filename
    _audio_ready = pyqtSignal(str)                 # internal: temp mp3 path

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

        # Audio state
        self._audio_loaded = False
        self._volume = 70       # 0-100
        self._muted = False
        self._temp_audio = None # path to temp mp3
        self._extract_token = 0 # cancels stale extractions

        # Clock-based sync state
        self._play_start_time = 0.0
        self._play_start_frame = 0

        if HAS_AUDIO:
            self._audio_ready.connect(self._on_audio_ready)

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
        self.display_label.setSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Expanding)
        self.display_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.display_label, 1)

        # Seek bar
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(self._seek_to_slider)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.seek_slider)

        # Time label
        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.btn_step_back = QPushButton("-5s")
        self.btn_step_back.clicked.connect(lambda: self._step(-5.0))
        ctrl_layout.addWidget(self.btn_step_back)

        self.btn_step_back_small = QPushButton("-1s")
        self.btn_step_back_small.clicked.connect(lambda: self._step(-1.0))
        ctrl_layout.addWidget(self.btn_step_back_small)

        self.btn_frame_back = QPushButton("< Frame")
        self.btn_frame_back.clicked.connect(lambda: self._step_frames(-1))
        ctrl_layout.addWidget(self.btn_frame_back)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        self.btn_frame_fwd = QPushButton("Frame >")
        self.btn_frame_fwd.clicked.connect(lambda: self._step_frames(1))
        ctrl_layout.addWidget(self.btn_frame_fwd)

        self.btn_step_fwd_small = QPushButton("+1s")
        self.btn_step_fwd_small.clicked.connect(lambda: self._step(1.0))
        ctrl_layout.addWidget(self.btn_step_fwd_small)

        self.btn_step_fwd = QPushButton("+5s")
        self.btn_step_fwd.clicked.connect(lambda: self._step(5.0))
        ctrl_layout.addWidget(self.btn_step_fwd)

        layout.addLayout(ctrl_layout)

        # Volume row (only shown when audio backend is available)
        if HAS_AUDIO:
            vol_layout = QHBoxLayout()
            vol_layout.addWidget(QLabel("Volume:"))
            self.volume_slider = QSlider(Qt.Horizontal)
            self.volume_slider.setRange(0, 100)
            self.volume_slider.setValue(self._volume)
            self.volume_slider.setMaximumWidth(150)
            self.volume_slider.valueChanged.connect(self._set_volume)
            vol_layout.addWidget(self.volume_slider)

            self.btn_mute = QPushButton("Mute")
            self.btn_mute.setCheckable(True)
            self.btn_mute.toggled.connect(self._toggle_mute)
            vol_layout.addWidget(self.btn_mute)

            self.audio_status_label = QLabel("No audio")
            self.audio_status_label.setStyleSheet("color: gray; font-size: 11px;")
            vol_layout.addWidget(self.audio_status_label)

            vol_layout.addStretch()
            layout.addLayout(vol_layout)

        # Mark button
        mark_layout = QHBoxLayout()
        self.btn_mark = QPushButton("Mark Experiment Start Here")
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

        # Per-video time frames status
        tf_layout = QHBoxLayout()
        self.tf_status_label = QLabel("Time frames: Global")
        self.tf_status_label.setStyleSheet("color: gray; font-size: 11px;")
        tf_layout.addWidget(self.tf_status_label, 1)
        self.btn_edit_tf = QPushButton("Edit time frames for this video")
        self.btn_edit_tf.setStyleSheet("font-size: 11px; padding: 3px 8px;")
        self.btn_edit_tf.clicked.connect(self._request_edit_timeframes)
        tf_layout.addWidget(self.btn_edit_tf)
        layout.addLayout(tf_layout)

    # --- Video directory / loading ---

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

        self.clear_mark_display()
        self.set_timeframes_status(False, 0)
        self.current_video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.display_label.setText("Cannot open video")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.seek_slider.setRange(0, max(0, self.total_frames - 1))
        self.current_frame_num = 0

        # Kick off background audio extraction
        self._audio_loaded = False
        if HAS_AUDIO:
            self._set_audio_status("Extracting audio...")
            self._extract_token += 1
            token = self._extract_token
            threading.Thread(
                target=self._extract_audio_thread,
                args=(os.path.abspath(path), token),
                daemon=True
            ).start()

        self._show_current_frame()

        # Notify main window so it can restore saved mark / TF status
        self.video_selected.emit(os.path.basename(path))

    # --- Audio extraction (background thread) ---

    def _extract_audio_thread(self, video_path: str, token: int):
        try:
            fd, tmp = tempfile.mkstemp(suffix='.mp3')
            os.close(fd)
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path,
                 '-vn', '-ar', '44100', '-ac', '2', '-q:a', '4', tmp],
                capture_output=True, timeout=120
            )
            if token != self._extract_token:
                # Stale — another video was loaded, discard
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                return
            if result.returncode == 0:
                self._audio_ready.emit(tmp)
            else:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                self._audio_ready.emit('')  # signal failure
        except Exception:
            self._audio_ready.emit('')

    def _on_audio_ready(self, tmp_path: str):
        # Runs in main thread via signal
        if not tmp_path:
            self._set_audio_status("No audio")
            return
        # Clean up previous temp file
        if self._temp_audio and self._temp_audio != tmp_path:
            try:
                os.unlink(self._temp_audio)
            except OSError:
                pass
        self._temp_audio = tmp_path
        try:
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.set_volume(
                0.0 if self._muted else self._volume / 100.0)
            self._audio_loaded = True
            self._set_audio_status("Audio ready")
        except Exception:
            self._audio_loaded = False
            self._set_audio_status("Audio error")

    def _set_audio_status(self, msg: str):
        if HAS_AUDIO and hasattr(self, 'audio_status_label'):
            self.audio_status_label.setText(msg)

    # --- Display ---

    def _show_current_frame(self):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
            self._update_ui_state()

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label_size = self.display_label.size()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled)

    def _update_ui_state(self):
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(self.current_frame_num)
        self.seek_slider.blockSignals(False)
        current_ms = (self.current_frame_num / self.fps) * 1000
        total_ms = (self.total_frames / self.fps) * 1000
        self.time_label.setText(
            f"{self._format_time(current_ms)} / {self._format_time(total_ms)}")

    def _format_time(self, ms: float) -> str:
        if ms < 0:
            ms = 0
        total_s = ms / 1000.0
        minutes = int(total_s // 60)
        seconds = total_s % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    # --- Playback controls ---

    def _toggle_play(self):
        if self.is_playing:
            self._stop()
        else:
            self._play()

    def _play(self):
        if self.cap is None:
            return
        self.is_playing = True
        self.btn_play.setText("Pause")
        interval = max(1, int(1000 / self.fps))

        if HAS_AUDIO and self._audio_loaded:
            start_s = self.current_frame_num / self.fps
            pygame.mixer.music.play(start=start_s)

        # Record wall-clock start *after* audio play() to minimise the gap
        self._play_start_time = time.perf_counter()
        self._play_start_frame = self.current_frame_num
        self.timer.start(interval)

    def _stop(self):
        self.is_playing = False
        self.btn_play.setText("Play")
        self.timer.stop()
        if HAS_AUDIO and self._audio_loaded:
            pygame.mixer.music.pause()

    def _next_frame(self):
        if self.cap is None:
            self._stop()
            return

        # Clock-based sync: compute which frame should be showing right now
        elapsed = time.perf_counter() - self._play_start_time
        expected_frame = int(self._play_start_frame + elapsed * self.fps)
        expected_frame = min(expected_frame, self.total_frames - 1)

        # If the timer fired late, skip ahead to catch up with the audio clock
        if expected_frame > self.current_frame_num + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)
            self.current_frame_num = expected_frame

        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num = int(
                self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._display_frame(frame)
            self._update_ui_state()
        else:
            self._stop()

    def _seek_to_slider(self, frame_num: int):
        self.current_frame_num = frame_num
        self._show_current_frame()

    def _on_slider_pressed(self):
        if self.is_playing:
            self.timer.stop()
            if HAS_AUDIO and self._audio_loaded:
                pygame.mixer.music.pause()

    def _on_slider_released(self):
        self.current_frame_num = self.seek_slider.value()
        self._show_current_frame()
        if self.is_playing:
            interval = max(1, int(1000 / self.fps))
            self.timer.start(interval)
            if HAS_AUDIO and self._audio_loaded:
                start_s = self.current_frame_num / self.fps
                pygame.mixer.music.play(start=start_s)

    def _step(self, seconds: float):
        if self.cap is None:
            return
        was_playing = self.is_playing
        if was_playing:
            self._stop()
        delta_frames = int(seconds * self.fps)
        self.current_frame_num = max(0, min(
            self.total_frames - 1, self.current_frame_num + delta_frames))
        self._show_current_frame()
        if was_playing:
            self._play()

    def _step_frames(self, n: int):
        if self.cap is None:
            return
        was_playing = self.is_playing
        if was_playing:
            self._stop()
        self.current_frame_num = max(0, min(
            self.total_frames - 1, self.current_frame_num + n))
        self._show_current_frame()

    def _set_volume(self, value: int):
        self._volume = value
        if HAS_AUDIO and self._audio_loaded and not self._muted:
            pygame.mixer.music.set_volume(value / 100.0)

    def _toggle_mute(self, muted: bool):
        self._muted = muted
        if HAS_AUDIO and self._audio_loaded:
            pygame.mixer.music.set_volume(
                0.0 if muted else self._volume / 100.0)
        self.btn_mute.setText("Unmute" if muted else "Mute")

    # --- Mark experiment start ---

    def _mark_start(self):
        if self.cap is None:
            return
        self.marked_ms = (self.current_frame_num / self.fps) * 1000
        self.mark_label.setText(
            f"Marked at {self._format_time(self.marked_ms)}")
        self.mark_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.experiment_marked.emit(self.marked_ms)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.frame_for_crop.emit(frame)

    def _request_edit_timeframes(self):
        name = self.get_current_video_name()
        if name:
            self.edit_video_timeframes_requested.emit(name)

    # --- Public setters / display helpers ---

    def set_mark_display(self, ms: float):
        """Show a saved mark (loaded from config) without emitting experiment_marked."""
        self.marked_ms = ms
        self.mark_label.setText(f"Marked at {self._format_time(ms)}")
        self.mark_label.setStyleSheet("color: #d32f2f; font-weight: bold;")

    def clear_mark_display(self):
        self.marked_ms = None
        self.mark_label.setText("No mark set")
        self.mark_label.setStyleSheet("color: gray;")

    def set_timeframes_status(self, has_custom: bool, phase_count: int):
        if has_custom:
            self.tf_status_label.setText(
                f"Time frames: Custom ({phase_count} phase(s))")
            self.tf_status_label.setStyleSheet(
                "color: #1565c0; font-size: 11px; font-weight: bold;")
        else:
            self.tf_status_label.setText("Time frames: Global")
            self.tf_status_label.setStyleSheet("color: gray; font-size: 11px;")

    # --- Public getters ---

    def get_marked_ms(self) -> float:
        return self.marked_ms if self.marked_ms is not None else 0.0

    def get_current_video_name(self) -> str:
        if self.current_video_path:
            return os.path.basename(self.current_video_path)
        return ""

    def get_current_frame(self):
        """Return the current frame as a BGR numpy array, or None."""
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        return frame if ret else None

    def cleanup(self):
        self._extract_token += 1  # invalidate any in-flight extraction
        self._stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if HAS_AUDIO:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        if self._temp_audio:
            try:
                os.unlink(self._temp_audio)
            except OSError:
                pass
            self._temp_audio = None
