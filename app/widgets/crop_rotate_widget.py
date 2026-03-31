import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QSizePolicy
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
        self.rotated_frame = None
        self.rotation_angle = 0.0
        self.pixmap_item = None
        self.crop_rect = None
        self.scale_factor = 1.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Graphics view for the frame
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(
            self.view.renderHints())
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.view, 1)

        # Rotation control
        rot_layout = QHBoxLayout()
        rot_layout.addWidget(QLabel("Rotation:"))
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-1800, 1800)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_slider.setTickInterval(450)
        self.rotation_slider.valueChanged.connect(self._on_rotation_changed)
        rot_layout.addWidget(self.rotation_slider)
        self.rotation_label = QLabel("0.0°")
        rot_layout.addWidget(self.rotation_label)
        layout.addLayout(rot_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Crop")
        self.btn_reset.clicked.connect(self._reset_crop)
        btn_layout.addWidget(self.btn_reset)

        self.btn_apply = QPushButton("Apply Crop Region")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #388e3c; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2e7d32; }")
        self.btn_apply.clicked.connect(self._apply_crop)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)

        self.status_label = QLabel("Load a video and mark the experiment start to set crop region")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

    def set_frame(self, frame: np.ndarray):
        self.original_frame = frame.copy()
        self.rotation_slider.setValue(0)
        self._update_display()

    def _on_rotation_changed(self, value):
        self.rotation_angle = value / 10.0
        self.rotation_label.setText(f"{self.rotation_angle:.1f}°")
        self._update_display()

    def _update_display(self):
        if self.original_frame is None:
            return

        self.scene.clear()
        self.crop_rect = None

        if self.rotation_angle != 0:
            self.rotated_frame = self._rotate(self.original_frame,
                                               self.rotation_angle)
        else:
            self.rotated_frame = self.original_frame.copy()

        rgb = cv2.cvtColor(self.rotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, w, h)

        # Add default crop rect (center 60% of frame)
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.6), int(h * 0.6)
        rx, ry = cx - rw // 2, cy - rh // 2
        self.crop_rect = ResizableRect(rx, ry, rw, rh)
        self.scene.addItem(self.crop_rect)

        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.status_label.setText(
            "Drag the green rectangle to select the froth region. "
            "Drag corners to resize.")

    def _rotate(self, frame: np.ndarray, angle: float) -> np.ndarray:
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

    def _reset_crop(self):
        self._update_display()

    def _apply_crop(self):
        if self.crop_rect is None:
            return

        # Get rect in scene coordinates
        scene_rect = self.crop_rect.mapRectToScene(self.crop_rect.rect())
        x = int(scene_rect.x())
        y = int(scene_rect.y())
        w = int(scene_rect.width())
        h = int(scene_rect.height())

        region = CropRegion(
            x=max(0, x),
            y=max(0, y),
            w=max(1, w),
            h=max(1, h),
            rotation_angle=self.rotation_angle
        )
        self.crop_applied.emit(region)
        self.status_label.setText(
            f"Crop region set: ({region.x}, {region.y}) "
            f"{region.w}x{region.h}, rotation {region.rotation_angle:.1f}°")
        self.status_label.setStyleSheet("color: #388e3c; font-weight: bold;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene.items():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
