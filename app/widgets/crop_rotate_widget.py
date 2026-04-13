import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QSizePolicy, QGroupBox
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QBrush

from ..core.models import CropRegion


class ResizableRect(QGraphicsRectItem):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        pen = QPen(QColor(0, 255, 0), 2)
        self.setPen(pen)
        brush = QBrush(QColor(0, 255, 0, 40))
        self.setBrush(brush)
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self._resize_handle_size = 12
        self._resizing = False
        self._resize_corner = None

    def contains_handle(self, pos):
        r = self.rect()
        s = self._resize_handle_size
        corners = {
            'tl': QRectF(r.x(), r.y(), s, s),
            'tr': QRectF(r.right() - s, r.y(), s, s),
            'bl': QRectF(r.x(), r.bottom() - s, s, s),
            'br': QRectF(r.right() - s, r.bottom() - s, s, s),
        }
        for name, rect in corners.items():
            if rect.contains(pos):
                return name
        return None

    def mousePressEvent(self, event):
        corner = self.contains_handle(event.pos())
        if corner:
            self._resizing = True
            self._resize_corner = corner
            self._start_pos = event.pos()
            self._start_rect = QRectF(self.rect())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            delta = event.pos() - self._start_pos
            r = QRectF(self._start_rect)
            if 'r' in self._resize_corner:
                r.setRight(r.right() + delta.x())
            if 'l' in self._resize_corner:
                r.setLeft(r.left() + delta.x())
            if 'b' in self._resize_corner:
                r.setBottom(r.bottom() + delta.y())
            if 't' in self._resize_corner:
                r.setTop(r.top() + delta.y())
            if r.width() > 20 and r.height() > 20:
                self.setRect(r)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._resize_corner = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CropRotateWidget(QWidget):
    crop_applied = pyqtSignal(CropRegion)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_frame = None
        self.transformed_frame = None
        self.rotation_angle = 0.0
        self.perspective_x = 0.0
        self.perspective_y = 0.0
        self.pixmap_item = None
        self.crop_rect = None

        # Time-slider state
        self._cap = None
        self._fps = 30.0
        self._total_frames = 0

        # When True, _update_display resets the rect to the default centre box.
        # Set by set_frame(); cleared after the first _update_display call.
        self._reset_crop_rect = False

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Graphics view for the frame
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.view, 1)

        # ── Time-preview slider ──────────────────────────────────────────
        time_group = QGroupBox("Preview Frame")
        time_layout = QVBoxLayout(time_group)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._on_time_slider_changed)
        time_layout.addWidget(self.time_slider)

        self.time_label = QLabel("--:---.--- / --:---.---")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 11px; color: gray;")
        time_layout.addWidget(self.time_label)

        layout.addWidget(time_group)

        # ── Transform controls ───────────────────────────────────────────
        transform_group = QGroupBox("Transform")
        transform_layout = QVBoxLayout(transform_group)

        rot_layout = QHBoxLayout()
        rot_layout.addWidget(QLabel("Rotation:"))
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-150, 150)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_slider.setTickInterval(50)
        self.rotation_slider.valueChanged.connect(self._on_transform_changed)
        rot_layout.addWidget(self.rotation_slider)
        self.rotation_label = QLabel("0.0°")
        self.rotation_label.setMinimumWidth(45)
        rot_layout.addWidget(self.rotation_label)
        transform_layout.addLayout(rot_layout)

        px_layout = QHBoxLayout()
        px_layout.addWidget(QLabel("Tilt L/R:"))
        self.persp_x_slider = QSlider(Qt.Horizontal)
        self.persp_x_slider.setRange(-300, 300)
        self.persp_x_slider.setValue(0)
        self.persp_x_slider.setTickPosition(QSlider.TicksBelow)
        self.persp_x_slider.setTickInterval(100)
        self.persp_x_slider.valueChanged.connect(self._on_transform_changed)
        px_layout.addWidget(self.persp_x_slider)
        self.persp_x_label = QLabel("0.0°")
        self.persp_x_label.setMinimumWidth(45)
        px_layout.addWidget(self.persp_x_label)
        transform_layout.addLayout(px_layout)

        py_layout = QHBoxLayout()
        py_layout.addWidget(QLabel("Tilt F/B:"))
        self.persp_y_slider = QSlider(Qt.Horizontal)
        self.persp_y_slider.setRange(-300, 300)
        self.persp_y_slider.setValue(0)
        self.persp_y_slider.setTickPosition(QSlider.TicksBelow)
        self.persp_y_slider.setTickInterval(100)
        self.persp_y_slider.valueChanged.connect(self._on_transform_changed)
        py_layout.addWidget(self.persp_y_slider)
        self.persp_y_label = QLabel("0.0°")
        self.persp_y_label.setMinimumWidth(45)
        py_layout.addWidget(self.persp_y_label)
        transform_layout.addLayout(py_layout)

        layout.addWidget(transform_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset All")
        self.btn_reset.clicked.connect(self._reset_all)
        btn_layout.addWidget(self.btn_reset)

        self.btn_apply = QPushButton("Apply Crop Region")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #388e3c; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2e7d32; }")
        self.btn_apply.clicked.connect(self._apply_crop)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)

        self.status_label = QLabel(
            "Load a video and mark the experiment start to set crop region")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------ #
    # Video loading for the time-preview slider
    # ------------------------------------------------------------------ #

    def set_video(self, path: str, fps: float, total_frames: int,
                  start_ms: float = 0.0):
        """Load video for the preview slider. Called when a video is selected."""
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        self._fps = fps if fps > 0 else 30.0
        self._total_frames = total_frames

        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, max(0, total_frames - 1))
        start_frame = int(start_ms / 1000.0 * self._fps)
        start_frame = max(0, min(start_frame, max(0, total_frames - 1)))
        self.time_slider.setValue(start_frame)
        self.time_slider.setEnabled(True)
        self.time_slider.blockSignals(False)
        self._update_time_label(start_frame)

    def _on_time_slider_changed(self, frame_num: int):
        if self._cap is None or self.original_frame is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._cap.read()
        if ret:
            self.original_frame = frame  # update without resetting rect/sliders
            self._update_display()
        self._update_time_label(frame_num)

    def _update_time_label(self, frame_num: int):
        if self._fps > 0 and self._total_frames > 0:
            self.time_label.setText(
                f"{self._fmt_ms(frame_num / self._fps * 1000)} / "
                f"{self._fmt_ms(self._total_frames / self._fps * 1000)}")

    @staticmethod
    def _fmt_ms(ms: float) -> str:
        total_s = max(0.0, ms) / 1000.0
        m = int(total_s // 60)
        s = total_s % 60
        return f"{m:02d}:{s:06.3f}"

    # ------------------------------------------------------------------ #
    # Frame loading
    # ------------------------------------------------------------------ #

    def set_frame(self, frame: np.ndarray):
        """Set a new frame and reset transforms + crop rect to defaults."""
        self.original_frame = frame.copy()
        self._reset_crop_rect = True  # force default rect on next _update_display

        for slider in (self.rotation_slider, self.persp_x_slider,
                       self.persp_y_slider):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.rotation_angle = 0.0
        self.perspective_x = 0.0
        self.perspective_y = 0.0
        self.rotation_label.setText("0.0°")
        self.persp_x_label.setText("0.0°")
        self.persp_y_label.setText("0.0°")

        self._update_display()

        # Sync time slider to this frame (without seeking — frame was pushed in)
        if self._fps > 0 and self._total_frames > 0 and self._cap:
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_num = int(pos_ms / 1000.0 * self._fps)
            frame_num = max(0, min(frame_num, self._total_frames - 1))
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(frame_num)
            self.time_slider.blockSignals(False)
            self._update_time_label(frame_num)

        self.status_label.setText(
            "Drag the green rectangle to select the froth region. "
            "Drag corners to resize.")

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def _on_transform_changed(self):
        self.rotation_angle = self.rotation_slider.value() / 10.0
        self.perspective_x = self.persp_x_slider.value() / 10.0
        self.perspective_y = self.persp_y_slider.value() / 10.0
        self.rotation_label.setText(f"{self.rotation_angle:.1f}°")
        self.persp_x_label.setText(f"{self.perspective_x:.1f}°")
        self.persp_y_label.setText(f"{self.perspective_y:.1f}°")
        self._update_display()

    def _update_display(self):
        if self.original_frame is None:
            return

        # Save current crop rect position — unless we're doing a full reset
        saved_rect = None
        if not self._reset_crop_rect and self.crop_rect is not None:
            saved_rect = self.crop_rect.mapRectToScene(self.crop_rect.rect())
        self._reset_crop_rect = False

        self.scene.clear()
        self.crop_rect = None

        # Apply perspective then rotation
        frame = self.original_frame.copy()
        if self.perspective_x != 0 or self.perspective_y != 0:
            frame = self._apply_perspective(
                frame, self.perspective_x, self.perspective_y)
        if self.rotation_angle != 0:
            frame = self._rotate(frame, self.rotation_angle)
        self.transformed_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.pixmap_item = self.scene.addPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)

        # Restore saved rect or use default centre 60%
        if saved_rect is not None and saved_rect.width() > 20 and \
                saved_rect.height() > 20:
            self.crop_rect = ResizableRect(
                saved_rect.x(), saved_rect.y(),
                saved_rect.width(), saved_rect.height())
        else:
            cx, cy = w // 2, h // 2
            rw, rh = 342, 228
            self.crop_rect = ResizableRect(
                cx - rw // 2, cy - rh // 2, rw, rh)

        self.scene.addItem(self.crop_rect)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_perspective(frame, angle_x, angle_y):
        h, w = frame.shape[:2]
        dx = np.tan(np.radians(angle_x)) * w * 0.3
        dy = np.tan(np.radians(angle_y)) * h * 0.3
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [0 + dx,  0 + dy],
            [w - dx,  0 - dy],
            [w + dx,  h + dy],
            [0 - dx,  h - dy],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (w, h))

    @staticmethod
    def _rotate(frame, angle):
        h, w = frame.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(frame, M, (new_w, new_h))

    def _reset_all(self):
        for slider in (self.rotation_slider, self.persp_x_slider,
                       self.persp_y_slider):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.rotation_angle = 0.0
        self.perspective_x = 0.0
        self.perspective_y = 0.0
        self.rotation_label.setText("0.0°")
        self.persp_x_label.setText("0.0°")
        self.persp_y_label.setText("0.0°")
        self._update_display()

    def _apply_crop(self):
        if self.crop_rect is None:
            return
        scene_rect = self.crop_rect.mapRectToScene(self.crop_rect.rect())
        x = int(scene_rect.x())
        y = int(scene_rect.y())
        w = int(scene_rect.width())
        h = int(scene_rect.height())
        region = CropRegion(
            x=max(0, x), y=max(0, y),
            w=max(1, w), h=max(1, h),
            rotation_angle=self.rotation_angle,
            perspective_x=self.perspective_x,
            perspective_y=self.perspective_y,
        )
        self.crop_applied.emit(region)
        parts = [f"({region.x}, {region.y}) {region.w}×{region.h}"]
        if region.rotation_angle:
            parts.append(f"rot {region.rotation_angle:.1f}°")
        if region.perspective_x or region.perspective_y:
            parts.append(
                f"persp ({region.perspective_x:.1f}°, "
                f"{region.perspective_y:.1f}°)")
        self.status_label.setText("Crop region set: " + ", ".join(parts))
        self.status_label.setStyleSheet("color: #388e3c; font-weight: bold;")

    def load_crop_region(self, region: CropRegion):
        """Restore a previously saved crop region into the widget."""
        if self.original_frame is None:
            return
        for slider, val in [
            (self.rotation_slider, int(region.rotation_angle * 10)),
            (self.persp_x_slider,  int(region.perspective_x  * 10)),
            (self.persp_y_slider,  int(region.perspective_y  * 10)),
        ]:
            slider.blockSignals(True)
            slider.setValue(val)
            slider.blockSignals(False)
        self.rotation_angle = region.rotation_angle
        self.perspective_x  = region.perspective_x
        self.perspective_y  = region.perspective_y
        self.rotation_label.setText(f"{self.rotation_angle:.1f}°")
        self.persp_x_label.setText(f"{self.perspective_x:.1f}°")
        self.persp_y_label.setText(f"{self.perspective_y:.1f}°")
        self._reset_crop_rect = True  # rebuild scene with transforms
        self._update_display()
        if self.crop_rect is not None:
            self.crop_rect.setRect(
                QRectF(region.x, region.y, region.w, region.h))
        parts = [f"({region.x}, {region.y}) {region.w}×{region.h}"]
        if region.rotation_angle:
            parts.append(f"rot {region.rotation_angle:.1f}°")
        if region.perspective_x or region.perspective_y:
            parts.append(
                f"persp ({region.perspective_x:.1f}°, "
                f"{region.perspective_y:.1f}°)")
        self.status_label.setText("Crop loaded: " + ", ".join(parts))
        self.status_label.setStyleSheet("color: #1565c0; font-weight: bold;")

    def cleanup(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene.items():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
