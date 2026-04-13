from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout,
)

from ..core.models import TimeFrame
from ..widgets.time_frame_editor import TimeFrameEditor


class VideoTimeFramesDialog(QDialog):
    """Edit per-video time frame overrides.

    Result can be either a custom list of TimeFrames (OK pressed)
    or None (Use Global pressed — clears the override).
    """

    def __init__(self, video_name: str,
                 custom_tfs: Optional[List[TimeFrame]],
                 global_tfs: List[TimeFrame],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Time Frames — {video_name}")
        self.setMinimumWidth(640)
        self._result_tfs: Optional[List[TimeFrame]] = ...  # sentinel

        layout = QVBoxLayout(self)

        status = ("Using custom time frames." if custom_tfs
                  else "Currently using global time frames.")
        layout.addWidget(QLabel(status))
        hint = QLabel(
            "Edit below to override for this video, or click "
            "\"Use Global\" to remove any custom override.")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.editor = TimeFrameEditor()
        self.editor.set_time_frames(custom_tfs if custom_tfs else global_tfs)
        layout.addWidget(self.editor)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to Global")
        btn_reset.setToolTip("Repopulate editor with global time frames")
        btn_reset.clicked.connect(
            lambda: self.editor.set_time_frames(global_tfs))
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        box = QDialogButtonBox()
        btn_ok = box.addButton("Save Custom", QDialogButtonBox.AcceptRole)
        btn_global = box.addButton("Use Global", QDialogButtonBox.ResetRole)
        btn_cancel = box.addButton(QDialogButtonBox.Cancel)

        btn_ok.clicked.connect(self._on_save_custom)
        btn_global.clicked.connect(self._on_use_global)
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(box)

    def _on_save_custom(self):
        self._result_tfs = self.editor.get_time_frames()
        self.accept()

    def _on_use_global(self):
        self._result_tfs = None
        self.accept()

    def get_result(self) -> Optional[List[TimeFrame]]:
        """None = use global.  List = custom override."""
        return self._result_tfs
