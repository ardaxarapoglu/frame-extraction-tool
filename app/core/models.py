from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class TimeFrame:
    name: str = "Phase1"
    duration_seconds: float = 30.0
    num_frames: int = 5
    naming_scheme: str = "{video}_{name}_{index:03d}"


@dataclass
class CropRegion:
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 100
    rotation_angle: float = 0.0
    # Perspective tilt correction (degrees, applied before rotation/crop)
    perspective_x: float = 0.0  # left-right tilt
    perspective_y: float = 0.0  # forward-back tilt


@dataclass
class ProjectConfig:
    video_directory: str = ""
    output_directory: str = ""
    time_frames: List[TimeFrame] = field(default_factory=list)
    experiment_start_ms: float = 0.0
    crop_region: Optional[CropRegion] = None
    obstruction_enabled: bool = True
    obstruction_sensitivity: float = 0.35
    per_video_start: bool = False
    video_start_marks: dict = field(default_factory=dict)
