import os
from typing import Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
import cv2
from .scene_detector import detect_scenes
from .speech_to_text import transcribe_video
from .visual_describer import keyframe_objects
from ..llm.gemini_client import llm_summarize_scene
from ..db.repository import create_video, add_highlight
from ..db.database import SessionLocal
from ..utils.time_utils import sec_to_timestamp

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

    def process(self, video_path: str, scene_threshold: int = 5) -> Dict:
        filename = os.path.basename(video_path)
        duration = _video_duration_seconds(video_path)
        video = create_video(self.db, filename=filename, duration_sec=duration)
        scenes = detect_scenes(video_path, threshold=scene_threshold)
        transcript = transcribe_video(video_path)

        def text_for_scene(start: float, end: float) -> str:
            parts = [seg["text"] for seg in transcript if seg["start"] < end and seg["end"] > start]
            return " ".join(parts).strip()

        highlights = []
        for (start, end) in tqdm(scenes, desc=f"Scenes in {filename}"):
            vis = keyframe_objects(video_path, (start, end))
            speech = text_for_scene(start, end)
            scene_payload = {
                "filename": filename,
                "start_sec": round(start,3), "end_sec": round(end,3),
                "start_ts": sec_to_timestamp(start), "end_ts": sec_to_timestamp(end),
                "objects": vis.get("objects", []),
                "motion_score": round(float(vis.get("motion",0.0)),2),
                "transcript_excerpt": speech[:500],
                "hints": "Mark as highlight if something noteworthy happens (explosion, crowd cheer, person speaking clearly, high motion, unusual objects)."
            }
            verdict = llm_summarize_scene(scene_payload)
            if verdict.get("is_highlight"):
                desc = f"Objects: {', '.join(scene_payload['objects'])}; Motion: {scene_payload['motion_score']}; Speech: {scene_payload['transcript_excerpt'][:140]}"
                summary = verdict.get("summary") or verdict.get("title") or "Highlight"
                vec = self.embedder.encode(summary).tolist()
                add_highlight(self.db, video_id=video.id, start_sec=start, end_sec=end,
                              description=desc, summary=summary, embedding=vec)
                highlights.append({"start": start, "end": end, "summary": summary})
        return {"video_id": str(video.id), "filename": filename, "scenes": len(scenes), "highlights": highlights}
