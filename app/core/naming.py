import os


def generate_filename(scheme: str, video_name: str, timeframe_name: str,
                      index: int, timestamp_ms: float = 0.0) -> str:
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
    if not name.lower().endswith(".png"):
        name += ".png"
    return name
