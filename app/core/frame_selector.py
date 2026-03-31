import numpy as np
from typing import List


def select_frames(good_indices: List[int], target_count: int) -> List[int]:
    if not good_indices:
        return []
    if len(good_indices) <= target_count:
        return list(good_indices)
    positions = np.linspace(0, len(good_indices) - 1, target_count)
    positions = np.round(positions).astype(int)
    return [good_indices[i] for i in positions]
