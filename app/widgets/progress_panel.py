from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QProgressBar,
    QTextEdit, QPushButton, QLabel
)
from PyQt5.QtCore import pyqtSignal


class ProgressPanel(QWidget):
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self.status_label)
        header.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        header.addWidget(self.btn_cancel)
        layout.addLayout(header)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: Consolas, monospace; font-size: 11px; }")
        layout.addWidget(self.log_text)

    def set_progress(self, value: int, message: str):
        if value > 0:
            self.progress_bar.setValue(value)
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())
        self.status_label.setText(message[:80])

    def set_processing(self, active: bool):
        self.btn_cancel.setEnabled(active)
        if active:
            self.status_label.setText("Processing...")
            self.progress_bar.setValue(0)
        else:
            if self.progress_bar.value() >= 100:
                self.status_label.setText("Complete")
            else:
                self.status_label.setText("Ready")

    def clear_log(self):
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
