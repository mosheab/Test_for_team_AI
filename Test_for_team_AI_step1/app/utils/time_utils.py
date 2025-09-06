def sec_to_timestamp(sec: float) -> str:
    sec = max(0, float(sec))
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}" if h else f"{m:02d}:{s:06.3f}"
