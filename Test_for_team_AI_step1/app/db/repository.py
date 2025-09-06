from typing import List
from sqlalchemy.orm import Session
from .models import Video, Highlight

def create_video(db: Session, filename: str, duration_sec: float | None) -> Video:
    v = Video(filename=filename, duration_sec=duration_sec)
    db.add(v); db.commit(); db.refresh(v); return v

def add_highlight(db: Session, video_id, start_sec: float, end_sec: float, description: str, summary: str, embedding):
    h = Highlight(video_id=video_id, start_sec=start_sec, end_sec=end_sec, description=description, summary=summary, embedding=embedding)
    db.add(h); db.commit(); db.refresh(h); return h

def list_highlights(db: Session, video_id) -> List[Highlight]:
    return db.query(Highlight).filter(Highlight.video_id == video_id).order_by(Highlight.start_sec).all()
