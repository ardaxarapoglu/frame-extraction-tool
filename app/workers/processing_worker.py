from PyQt5.QtCore import QThread, pyqtSignal
from ..core.models import ProjectConfig
from ..core.video_processor import VideoProcessor
import copy


class ProcessingWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    finished_processing = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, config: ProjectConfig, single_video=None,
                 selected_videos=None, parent=None):
        super().__init__(parent)
        self.config = copy.deepcopy(config)
        self.single_video = single_video
        self.selected_videos = selected_videos  # List[str] or None
        self._cancelled = False

    def run(self):
        try:
            processor = VideoProcessor(
                self.config,
                progress_callback=self._on_progress,
                cancel_check=lambda: self._cancelled,
            )
            if self.single_video:
                processor.process_single(self.single_video)
            elif self.selected_videos is not None:
                processor.process_selected(self.selected_videos)
            else:
                processor.process_all()
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished_processing.emit()

    def _on_progress(self, pct: int, message: str):
        self.progress_updated.emit(pct, message)

    def cancel(self):
        self._cancelled = True
