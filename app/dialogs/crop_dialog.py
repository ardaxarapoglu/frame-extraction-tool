import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox
from ..widgets.crop_rotate_widget import CropRotateWidget
from ..core.models import CropRegion


class CropDialog(QDialog):
    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Crop & Rotation")
        self.setMinimumSize(900, 700)
        self.crop_region = None

        layout = QVBoxLayout(self)
        self.crop_widget = CropRotateWidget()
        self.crop_widget.set_frame(frame)
        self.crop_widget.crop_applied.connect(self._on_crop_applied)
        layout.addWidget(self.crop_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_crop_applied(self, region: CropRegion):
        self.crop_region = region

    def get_crop_region(self) -> CropRegion:
        return self.crop_region
