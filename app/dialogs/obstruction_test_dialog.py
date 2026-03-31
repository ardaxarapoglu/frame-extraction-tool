import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QScrollArea, QWidget, QGridLayout,
    QSpinBox, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont

from ..core.obstruction_detector import ObstructionDetector


class ObstructionTestDialog(QDialog):
    def __init__(self, frames, sensitivity=0.35, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Obstruction Detection")
        self.setMinimumSize(900, 650)
        self.frames = frames  # list of cropped BGR numpy arrays
        self.sensitivity = sensitivity
        self.results = []  # (is_obstructed, score) per frame
        self._setup_ui()
        self._run_detection()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info
        info = QLabel(f"Testing on {len(self.frames)} frames. "
                      "Red border = detected as obstructed. "
                      "Adjust sensitivity and re-test.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Sensitivity control
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

        self.btn_retest = QPushButton("Re-test")
        self.btn_retest.setStyleSheet(
            "QPushButton { background-color: #f57c00; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e65100; }")
        self.btn_retest.clicked.connect(self._run_detection)
        ctrl_layout.addWidget(self.btn_retest)

        layout.addLayout(ctrl_layout)

        # Thumbnail columns
        cols_layout = QHBoxLayout()
        cols_layout.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 10)
        self.cols_spin.setValue(5)
        self.cols_spin.valueChanged.connect(self._update_grid)
        cols_layout.addWidget(self.cols_spin)
        cols_layout.addStretch()

        # Stats
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("font-weight: bold;")
        cols_layout.addWidget(self.stats_label)
        layout.addLayout(cols_layout)

        # Scroll area with grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll, 1)

        # Buttons
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

    def _run_detection(self):
        self.sensitivity = self.sensitivity_slider.value() / 100.0
        detector = ObstructionDetector(sensitivity=self.sensitivity)
        detector.build_reference(self.frames)
        self.results = []
        for frame in self.frames:
            is_bad, score = detector.is_obstructed(frame)
            self.results.append((is_bad, score))
        self._update_grid()

    def _update_grid(self):
        # Clear existing
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        cols = self.cols_spin.value()
        thumb_size = 160
        good_count = 0
        bad_count = 0

        for i, (frame, (is_bad, score)) in enumerate(
                zip(self.frames, self.results)):
            if is_bad:
                bad_count += 1
            else:
                good_count += 1

            # Create thumbnail
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w_img, ch = rgb.shape
            qimg = QImage(rgb.data, w_img, h, ch * w_img, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                thumb_size, thumb_size, Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

            # Draw border and label
            bordered = QPixmap(pixmap.width() + 6, pixmap.height() + 22)
            border_color = QColor(220, 40, 40) if is_bad else QColor(40, 180, 40)
            bordered.fill(border_color)

            painter = QPainter(bordered)
            painter.drawPixmap(3, 3, pixmap)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 8))
            label_text = f"#{i} {'BAD' if is_bad else 'OK'} ({score:.3f})"
            painter.drawText(3, bordered.height() - 5, label_text)
            painter.end()

            lbl = QLabel()
            lbl.setPixmap(bordered)
            lbl.setAlignment(Qt.AlignCenter)
            row, col = divmod(i, cols)
            self.grid_layout.addWidget(lbl, row, col)

        self.stats_label.setText(
            f"Good: {good_count}  |  Obstructed: {bad_count}  |  "
            f"Total: {len(self.frames)}")

    def get_sensitivity(self) -> float:
        return self.sensitivity_slider.value() / 100.0
