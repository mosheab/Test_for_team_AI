import os
from typing import List, Dict, Any
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import av

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")


def has_audio(video_path: str) -> bool:
    try:
        container = av.open(video_path)
        return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return False

def transcribe_video(video_path: str) -> List[Dict[str, Any]]:
    if not has_audio(video_path):
        print(f"[INFO] No audio track in {video_path}, skipping transcription")
        return []

    try:
        audio = decode_audio(video_path, sampling_rate=16000)
    except Exception as e:
        print(f"[WARN] Failed to decode audio in {video_path}: {e}")
        return []

    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="int8")
    segments, _ = model.transcribe(audio)
    return [
        {"start": float(s.start), "end": float(s.end), "text": s.text}
        for s in segments
    ]
