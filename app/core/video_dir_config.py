import json
import os
from typing import Dict, List, Optional, Any

from .models import TimeFrame

CONFIG_FILENAME = "fe_config.json"


def _tfs_to_list(tfs: List[TimeFrame]) -> list:
    return [
        {"name": tf.name, "duration_seconds": tf.duration_seconds,
         "num_frames": tf.num_frames, "naming_scheme": tf.naming_scheme}
        for tf in tfs
    ]


def _list_to_tfs(data: list) -> List[TimeFrame]:
    return [
        TimeFrame(
            name=d.get("name", "Phase1"),
            duration_seconds=float(d.get("duration_seconds", 30.0)),
            num_frames=int(d.get("num_frames", 5)),
            naming_scheme=d.get("naming_scheme", "{video}_{name}_{index:03d}"),
        )
        for d in data
    ]


class VideoDirectoryConfig:
    """Reads/writes fe_config.json in the video directory.

    Persists global settings (output dir, obstruction, time frames, etc.)
    and per-video settings (start marks, custom time frames) so the user
    never has to re-enter them after a restart.
    """

    def __init__(self):
        self._directory: str = ""
        self._data: Dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return bool(self._directory)

    def load(self, directory: str):
        self._directory = directory
        path = os.path.join(directory, CONFIG_FILENAME)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def save(self):
        if not self._directory:
            return
        path = os.path.join(self._directory, CONFIG_FILENAME)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Generic key-value access for scalar settings
    # ------------------------------------------------------------------ #

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value

    # ------------------------------------------------------------------ #
    # Global time frames
    # ------------------------------------------------------------------ #

    def get_global_time_frames(self) -> List[TimeFrame]:
        data = self._data.get("global_time_frames", [])
        return _list_to_tfs(data) if data else []

    def set_global_time_frames(self, tfs: List[TimeFrame]):
        self._data["global_time_frames"] = _tfs_to_list(tfs)

    # ------------------------------------------------------------------ #
    # Per-video helpers
    # ------------------------------------------------------------------ #

    def _video_entry(self, video_name: str) -> dict:
        self._data.setdefault("videos", {}).setdefault(video_name, {})
        return self._data["videos"][video_name]

    def get_video_start_ms(self, video_name: str) -> Optional[float]:
        return self._data.get("videos", {}).get(video_name, {}).get("start_ms")

    def set_video_start_ms(self, video_name: str, ms: float):
        self._video_entry(video_name)["start_ms"] = ms

    def get_video_time_frames(self, video_name: str) -> Optional[List[TimeFrame]]:
        raw = self._data.get("videos", {}).get(video_name, {}).get(
            "custom_time_frames")
        return _list_to_tfs(raw) if raw is not None else None

    def set_video_time_frames(self, video_name: str,
                              tfs: Optional[List[TimeFrame]]):
        entry = self._video_entry(video_name)
        entry["custom_time_frames"] = _tfs_to_list(tfs) if tfs is not None else None

    def get_all_video_start_marks(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name, vdata in self._data.get("videos", {}).items():
            if "start_ms" in vdata:
                result[name] = vdata["start_ms"]
        return result

    def get_video_crop_region(self, video_name: str):
        from .models import CropRegion
        data = self._data.get("videos", {}).get(video_name, {}).get("crop_region")
        if data is None:
            return None
        return CropRegion(
            x=data.get("x", 0),
            y=data.get("y", 0),
            w=data.get("w", 100),
            h=data.get("h", 100),
            rotation_angle=data.get("rotation_angle", 0.0),
            perspective_x=data.get("perspective_x", 0.0),
            perspective_y=data.get("perspective_y", 0.0),
        )

    def set_video_crop_region(self, video_name: str, region):
        entry = self._video_entry(video_name)
        if region is None:
            entry["crop_region"] = None
        else:
            entry["crop_region"] = {
                "x": region.x, "y": region.y,
                "w": region.w, "h": region.h,
                "rotation_angle": region.rotation_angle,
                "perspective_x": region.perspective_x,
                "perspective_y": region.perspective_y,
            }

    def get_all_video_crop_regions(self) -> dict:
        """Return {video_filename: CropRegion} for every video that has one saved."""
        result = {}
        for name, vdata in self._data.get("videos", {}).items():
            if vdata.get("crop_region") is not None:
                region = self.get_video_crop_region(name)
                if region is not None:
                    result[name] = region
        return result

    def get_all_custom_time_frames(self) -> Dict[str, List[TimeFrame]]:
        result: Dict[str, List[TimeFrame]] = {}
        for name, vdata in self._data.get("videos", {}).items():
            raw = vdata.get("custom_time_frames")
            if raw is not None:
                result[name] = _list_to_tfs(raw)
        return result
