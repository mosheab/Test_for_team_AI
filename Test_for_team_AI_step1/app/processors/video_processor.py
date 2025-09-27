import os
from typing import Dict
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
import cv2
from ..llm.gemini_client import llm_summarize_video
from ..db.repository import create_video, add_highlight
from ..db.database import SessionLocal

def _video_duration_seconds(path: str) -> float | None:
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and frames > 0:
            return float(frames / fps)
        return None
    finally:
        cap.release()

class VideoProcessor:
    def __init__(self, db: Session | None = None):
        self.db = db or SessionLocal()
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def process(self, video_path: str, max_highlights: int = 5) -> Dict:
        filename = os.path.basename(video_path)
        duration = _video_duration_seconds(video_path)
        video = create_video(self.db, filename=filename, duration_sec=duration)
        highlights = llm_summarize_video(video_path, max_highlights)
        results = []

        for item in highlights:
            start = float(item.get("start_s", item.get("start", 0.0)))
            end   = float(item.get("end_s",   item.get("end",   start)))
            title = str(item.get("title", "")).strip()
            summary = str(item.get("summary", "")).strip()

            vec = self.embedder.encode(summary).tolist()
            add_highlight(
                self.db,
                video_id=video.id,
                start_sec=start,
                end_sec=end,
                title=title,
                summary=summary,
                embedding=vec,
            )

            results.append(
                {"start": start, "end": end, "title": title, "summary": summary}
            )

        return {
            "filename": filename,
            "highlights": results
        }
