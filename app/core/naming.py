import os


def generate_filename(scheme: str, video_name: str, timeframe_name: str,
                      index: int, timestamp_ms: float = 0.0,
                      rel_time_s: float = 0.0) -> str:
    base = os.path.splitext(video_name)[0]
    try:
        name = scheme.format(
            video=base,
            name=timeframe_name,
            index=index + 1,
            timestamp=int(timestamp_ms),
        )
    except (KeyError, IndexError, ValueError):
        name = f"{base}_{timeframe_name}_{index:03d}"
    # Strip extension so we can insert the relative timestamp before it
    if name.lower().endswith(".png"):
        name = name[:-4]
    name = f"{name}_{rel_time_s:.3f}s.png"
    return name
