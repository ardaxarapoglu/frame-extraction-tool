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


def select_frames_normal(good_indices: List[int], target_count: int) -> List[int]:
    """Select frames weighted toward the centre of the clip.

    Uses a normal distribution (mu=0.5, sigma=0.25 of the clip length)
    to build a probability mass function over the available frames, then
    draws *target_count* samples via inverse-CDF stratified sampling so
    the result is fully deterministic.
    """
    if not good_indices:
        return []
    if len(good_indices) <= target_count:
        return list(good_indices)

    n = len(good_indices)
    # Normalised position of each frame in [0, 1]
    pos = np.arange(n) / (n - 1) if n > 1 else np.array([0.5])

    # Normal PDF centred at 0.5; sigma=0.25 → weight at each edge is ~13.5%
    # of the peak, giving a clear but not extreme bell shape
    sigma = 0.25
    weights = np.exp(-0.5 * ((pos - 0.5) / sigma) ** 2)
    weights /= weights.sum()

    # Build CDF and sample target_count evenly-spaced quantiles from it
    # (stratified inverse-CDF sampling — deterministic, no randomness)
    cdf = np.cumsum(weights)
    quantiles = np.linspace(0.0, 1.0, target_count + 2)[1:-1]  # skip 0 and 1
    selected = []
    for q in quantiles:
        idx = int(np.searchsorted(cdf, q))
        idx = min(idx, n - 1)
        selected.append(good_indices[idx])

    return selected
