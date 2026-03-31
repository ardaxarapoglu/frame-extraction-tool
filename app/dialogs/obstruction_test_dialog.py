import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QScrollArea, QWidget, QGridLayout,
    QSpinBox, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont

from ..core.obstruction_detector import ObstructionDetector
from ..core.models import CropRegion


class ObstructionTestDialog(QDialog):
    def __init__(self, cap, crop_region, sensitivity=0.35, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Obstruction Detection")
        self.setMinimumSize(950, 700)
        self.cap = cap
        self.crop_region = crop_region
        self.sensitivity = sensitivity
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ref_frames = []
        self.test_frames = []
        self.results = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        info = QLabel(
            "1) Seek to clean froth and click 'Grab Reference Frames'\n"
            "2) Seek to where your hand/card appears and click "
            "'Grab Test Frames'\n"
            "3) Adjust sensitivity and click 'Run Test'")
        info.setWordWrap(True)
        info.setStyleSheet("padding: 4px; background: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        # Seek bar
        seek_layout = QHBoxLayout()
        seek_layout.addWidget(QLabel("Position:"))
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, max(0, self.total_frames - 1))
        self.seek_slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.seek_slider.valueChanged.connect(self._on_seek)
        seek_layout.addWidget(self.seek_slider, 1)
        self.time_label = QLabel("")
        seek_layout.addWidget(self.time_label)
        layout.addLayout(seek_layout)

        # Preview of current position
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(120)
        self.preview_label.setStyleSheet("background: black;")
        layout.addWidget(self.preview_label)

        # Grab buttons
        grab_layout = QHBoxLayout()

        self.btn_grab_ref = QPushButton("Grab Reference Frames (clean froth)")
        self.btn_grab_ref.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; "
            "font-weight: bold; padding: 6px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1565c0; }")
        self.btn_grab_ref.clicked.connect(self._grab_reference)
        grab_layout.addWidget(self.btn_grab_ref)

        self.ref_status = QLabel("No reference frames")
        self.ref_status.setStyleSheet("color: gray;")
        grab_layout.addWidget(self.ref_status)

        grab_layout.addSpacing(20)

        self.btn_grab_test = QPushButton("Grab Test Frames (with obstructions)")
        self.btn_grab_test.setStyleSheet(
            "QPushButton { background-color: #f57c00; color: white; "
            "font-weight: bold; padding: 6px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e65100; }")
        self.btn_grab_test.clicked.connect(self._grab_test)
        grab_layout.addWidget(self.btn_grab_test)

        self.test_status = QLabel("No test frames")
        self.test_status.setStyleSheet("color: gray;")
        grab_layout.addWidget(self.test_status)

        layout.addLayout(grab_layout)

        # Sensitivity + run
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(5, 90)
        self.sensitivity_slider.setValue(int(self.sensitivity * 100))
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(5)
        ctrl_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label = QLabel(f"{self.sensitivity:.2f}")
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v / 100:.2f}"))
        ctrl_layout.addWidget(self.sensitivity_label)

        self.btn_run = QPushButton("Run Test")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #388e3c; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2e7d32; }")
        self.btn_run.clicked.connect(self._run_detection)
        ctrl_layout.addWidget(self.btn_run)
        layout.addLayout(ctrl_layout)

        # Columns + stats
        cols_layout = QHBoxLayout()
        cols_layout.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 10)
        self.cols_spin.setValue(5)
        self.cols_spin.valueChanged.connect(self._update_grid)
        cols_layout.addWidget(self.cols_spin)
        cols_layout.addStretch()
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("font-weight: bold;")
        cols_layout.addWidget(self.stats_label)
        layout.addLayout(cols_layout)

        # Results grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll, 1)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_apply = QPushButton("Apply This Sensitivity")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #388e3c; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }")
        self.btn_apply.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_apply)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        # Show initial preview
        self._on_seek(self.seek_slider.value())

    def _on_seek(self, frame_num):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            frame = self._apply_crop(frame)
            self._show_preview(frame)
        time_s = frame_num / self.fps
        total_s = self.total_frames / self.fps
        self.time_label.setText(
            f"{time_s:.1f}s / {total_s:.1f}s")

    def _show_preview(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)

    def _grab_frames_at_current(self):
        pos = self.seek_slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        sample_count = int(self.fps * 3)
        raw = []
        for _ in range(sample_count):
            ret, frame = self.cap.read()
            if not ret:
                break
            raw.append(frame)
        # Crop and take every 3rd
        frames = []
        for frame in raw[::3]:
            frames.append(self._apply_crop(frame))
        return frames

    def _apply_crop(self, frame):
        crop = self.crop_region
        if crop is None:
            return frame
        if crop.perspective_x != 0 or crop.perspective_y != 0:
            h, w = frame.shape[:2]
            dx = np.tan(np.radians(crop.perspective_x)) * w * 0.3
            dy = np.tan(np.radians(crop.perspective_y)) * h * 0.3
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [0 + dx, 0 + dy], [w - dx, 0 - dy],
                [w + dx, h + dy], [0 - dx, h - dy],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            frame = cv2.warpPerspective(frame, M, (w, h))
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
        return frame[y:y + ch, x:x + cw]

    def _grab_reference(self):
        self.ref_frames = self._grab_frames_at_current()
        count = len(self.ref_frames)
        self.ref_status.setText(f"{count} reference frames grabbed")
        self.ref_status.setStyleSheet(
            "color: #1976d2; font-weight: bold;" if count > 0
            else "color: red;")

    def _grab_test(self):
        self.test_frames = self._grab_frames_at_current()
        count = len(self.test_frames)
        self.test_status.setText(f"{count} test frames grabbed")
        self.test_status.setStyleSheet(
            "color: #f57c00; font-weight: bold;" if count > 0
            else "color: red;")

    def _run_detection(self):
        if not self.ref_frames:
            self.stats_label.setText(
                "Grab reference frames first!")
            self.stats_label.setStyleSheet(
                "font-weight: bold; color: red;")
            return
        if not self.test_frames:
            self.stats_label.setText(
                "Grab test frames first!")
            self.stats_label.setStyleSheet(
                "font-weight: bold; color: red;")
            return

        self.sensitivity = self.sensitivity_slider.value() / 100.0
        detector = ObstructionDetector(sensitivity=self.sensitivity)
        detector.build_reference(self.ref_frames)

        # Test against all frames: reference (should be OK) + test (may be bad)
        all_frames = self.ref_frames + self.test_frames
        self.results = []
        for frame in all_frames:
            is_bad, score = detector.is_obstructed(frame)
            self.results.append((is_bad, score))
        self._all_display_frames = all_frames
        self._ref_count = len(self.ref_frames)
        self._update_grid()

    def _update_grid(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not hasattr(self, '_all_display_frames') or not self._all_display_frames:
            return

        cols = self.cols_spin.value()
        thumb_size = 150
        good_count = 0
        bad_count = 0
        ref_false_rejects = 0
        test_correct_rejects = 0

        for i, (frame, (is_bad, score)) in enumerate(
                zip(self._all_display_frames, self.results)):
            is_ref = i < self._ref_count
            if is_bad:
                bad_count += 1
                if is_ref:
                    ref_false_rejects += 1
                else:
                    test_correct_rejects += 1
            else:
                good_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w_img, ch = rgb.shape
            qimg = QImage(rgb.data, w_img, h, ch * w_img, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                thumb_size, thumb_size, Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

            bordered = QPixmap(pixmap.width() + 6, pixmap.height() + 26)
            if is_ref:
                border_color = (QColor(220, 40, 40) if is_bad
                                else QColor(40, 120, 220))
            else:
                border_color = (QColor(220, 40, 40) if is_bad
                                else QColor(40, 180, 40))
            bordered.fill(border_color)

            painter = QPainter(bordered)
            painter.drawPixmap(3, 3, pixmap)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 8))
            source = "REF" if is_ref else "TEST"
            status = "BAD" if is_bad else "OK"
            label_text = f"#{i} {source} {status} ({score:.3f})"
            painter.drawText(3, bordered.height() - 5, label_text)
            painter.end()

            lbl = QLabel()
            lbl.setPixmap(bordered)
            lbl.setAlignment(Qt.AlignCenter)
            row, col = divmod(i, cols)
            self.grid_layout.addWidget(lbl, row, col)

        test_total = len(self.test_frames)
        ref_total = len(self.ref_frames)
        self.stats_label.setText(
            f"Reference: {ref_total - ref_false_rejects}/{ref_total} OK "
            f"({ref_false_rejects} false rejects)  |  "
            f"Test: {test_correct_rejects}/{test_total} detected as obstructed")
        self.stats_label.setStyleSheet("font-weight: bold;")

    def get_sensitivity(self) -> float:
        return self.sensitivity_slider.value() / 100.0
