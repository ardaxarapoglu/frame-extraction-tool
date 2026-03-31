from PyQt5.QtCore import QThread, pyqtSignal
from ..core.models import ProjectConfig
from ..core.video_processor import VideoProcessor
import copy


class ProcessingWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    finished_processing = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, config: ProjectConfig, parent=None):
        super().__init__(parent)
        self.config = copy.deepcopy(config)
        self._cancelled = False

    def run(self):
        try:
            processor = VideoProcessor(
                self.config,
                progress_callback=self._on_progress,
                cancel_check=lambda: self._cancelled,
            )
            processor.process_all()
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished_processing.emit()

    def _on_progress(self, pct: int, message: str):
        self.progress_updated.emit(pct, message)

    def cancel(self):
        self._cancelled = True
