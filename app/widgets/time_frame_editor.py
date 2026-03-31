from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt
from typing import List
from ..core.models import TimeFrame


class TimeFrameEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Name", "Duration (s)", "Frames to Extract", "Naming Scheme"
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ Add Time Frame")
        self.btn_add.clicked.connect(self._add_row)
        btn_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("- Remove Selected")
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_layout.addWidget(self.btn_remove)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Add a default row
        self._add_row()

    def _add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        idx = row + 1
        self.table.setItem(row, 0, QTableWidgetItem(f"Phase{idx}"))
        self.table.setItem(row, 1, QTableWidgetItem("30"))
        self.table.setItem(row, 2, QTableWidgetItem("5"))
        self.table.setItem(row, 3, QTableWidgetItem(
            "{video}_{name}_{index:03d}"))

    def _remove_selected(self):
        rows = set(idx.row() for idx in self.table.selectedIndexes())
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def get_time_frames(self) -> List[TimeFrame]:
        frames = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0)
            dur = self.table.item(row, 1)
            num = self.table.item(row, 2)
            scheme = self.table.item(row, 3)
            if name and dur and num and scheme:
                try:
                    tf = TimeFrame(
                        name=name.text().strip(),
                        duration_seconds=float(dur.text().strip()),
                        num_frames=int(num.text().strip()),
                        naming_scheme=scheme.text().strip(),
                    )
                    frames.append(tf)
                except ValueError:
                    continue
        return frames

    def set_time_frames(self, time_frames: List[TimeFrame]):
        self.table.setRowCount(0)
        for tf in time_frames:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(tf.name))
            self.table.setItem(row, 1, QTableWidgetItem(
                str(tf.duration_seconds)))
            self.table.setItem(row, 2, QTableWidgetItem(str(tf.num_frames)))
            self.table.setItem(row, 3, QTableWidgetItem(tf.naming_scheme))
