import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QPlainTextEdit, QAbstractItemView, QSizePolicy
)
from PyQt5.QtCore import Qt


class DebugPanel(QWidget):
    def __init__(self, get_time_frames_fn=None, parent=None):
        super().__init__(parent)
        self._output_dir = ""
        self._get_time_frames = get_time_frames_fn or (lambda: [])
        self._scan_data = []  # list of (phase_name, video_base, [paths], spinbox)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        info = QLabel(
            "Workflow:\n"
            "1. Run processing to generate _unfiltered frames\n"
            "2. Manually copy your chosen frames into:\n"
            "       {output_dir}/{phase}/_manually-filtered/\n"
            "3. Click Scan, adjust frame counts, then Process."
        )
        info.setStyleSheet(
            "color: #555; font-size: 11px; background: #f5f5f5; "
            "padding: 8px; border-radius: 4px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Output dir display
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output dir:"))
        self.dir_label = QLabel("(not set)")
        self.dir_label.setStyleSheet("color: gray; font-size: 11px;")
        self.dir_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        dir_layout.addWidget(self.dir_label, 1)
        layout.addLayout(dir_layout)

        # Scan button
        self.btn_scan = QPushButton("Scan for Manually Filtered Frames")
        self.btn_scan.clicked.connect(self._scan)
        layout.addWidget(self.btn_scan)

        # Results table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Phase", "Video", "Available", "Select"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 1)

        # Process button
        self.btn_process = QPushButton("▶ Process Manually Filtered Frames")
        self.btn_process.setStyleSheet(
            "QPushButton { background-color: #7b1fa2; color: white; "
            "font-weight: bold; padding: 10px; font-size: 13px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background-color: #6a1b9a; }"
            "QPushButton:disabled { background-color: #90a4ae; }")
        self.btn_process.clicked.connect(self._process)
        layout.addWidget(self.btn_process)

        # Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.log, 1)

    # --- Public setters ---

    def set_output_dir(self, path: str):
        self._output_dir = path
        self.dir_label.setText(path or "(not set)")
        self.dir_label.setStyleSheet(
            "font-size: 11px; color: black;" if path else "font-size: 11px; color: gray;")

    # --- Scan ---

    def _scan(self):
        self.table.setRowCount(0)
        self._scan_data = []

        if not self._output_dir or not os.path.isdir(self._output_dir):
            self._log("Output directory not set or does not exist.")
            return

        time_frames = self._get_time_frames()
        tf_map = {tf.name: tf.num_frames for tf in time_frames}

        found_any = False
        # Structure: {output_dir}/{video_name}/{phase}/_manually-filtered/
        for video_name in sorted(os.listdir(self._output_dir)):
            video_path = os.path.join(self._output_dir, video_name)
            if not os.path.isdir(video_path) or video_name.startswith('_'):
                continue

            for phase_name in sorted(os.listdir(video_path)):
                phase_path = os.path.join(video_path, phase_name)
                if not os.path.isdir(phase_path) or phase_name.startswith('_'):
                    continue

                mf_path = os.path.join(phase_path, "_manually-filtered")
                if not os.path.isdir(mf_path):
                    continue

                img_files = sorted(
                    f for f in os.listdir(mf_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                )
                if not img_files:
                    self._log(
                        f"  {video_name}/{phase_name}/_manually-filtered/ "
                        f"is empty — skipping.")
                    continue

                paths = [os.path.join(mf_path, f) for f in img_files]
                default_n = min(tf_map.get(phase_name, 10), len(paths))

                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(phase_name))
                self.table.setItem(row, 1, QTableWidgetItem(video_name))
                self.table.setItem(row, 2, QTableWidgetItem(str(len(paths))))

                sb = QSpinBox()
                sb.setRange(1, len(paths))
                sb.setValue(default_n)
                self.table.setCellWidget(row, 3, sb)

                self._scan_data.append((phase_name, video_name, paths, sb))
                found_any = True

        if found_any:
            self._log(
                f"Scan complete — {self.table.rowCount()} "
                f"video/phase combination(s) found.")
        else:
            self._log(
                "No _manually-filtered folders found.\n"
                "Expected path: "
                "{output_dir}/{video_name}/{phase}/_manually-filtered/")

    # --- Process ---

    def _process(self):
        if not self._scan_data:
            self._log("Nothing to process — run Scan first.")
            return

        self._log("--- Processing ---")
        total_saved = 0

        for phase_name, video_base, frame_paths, sb in self._scan_data:
            n_want = sb.value()
            n_avail = len(frame_paths)

            if n_avail == 0:
                self._log(f"  [{video_base} / {phase_name}] No frames, skipping.")
                continue

            # Select n_want evenly spaced frames starting from index 0.
            # interval = n_avail / n_want; indices: 0, interval, 2*interval, ...
            interval = n_avail / n_want
            selected = [
                frame_paths[min(int(i * interval), n_avail - 1)]
                for i in range(n_want)
            ]

            out_dir = os.path.join(self._output_dir, video_base, phase_name)
            os.makedirs(out_dir, exist_ok=True)

            self._log(
                f"  [{video_base} / {phase_name}] "
                f"{n_avail} available → {len(selected)} selected "
                f"(every {interval:.1f} frames) → {out_dir}")

            for i, src in enumerate(selected):
                dst = os.path.join(out_dir, f"{i + 1:03d}_{os.path.basename(src)}")
                shutil.copy2(src, dst)
                total_saved += 1

        self._log(f"--- Done. {total_saved} frame(s) saved. ---")

    # --- Helpers ---

    def _log(self, msg: str):
        self.log.appendPlainText(msg)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())
