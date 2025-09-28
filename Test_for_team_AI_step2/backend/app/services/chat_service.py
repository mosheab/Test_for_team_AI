from typing import Dict, Any
from sqlalchemy.orm import Session
from ..repositories.highlights_repository import search_by_vector
from .embeddings import embed_text
from ..core.config import settings

def sec_to_timestamp(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}" if h else f"{m:02d}:{s:06.3f}"

def answer_query(db: Session, query: str, top_k: int | None = None) -> Dict[str, Any]:
    top_k = top_k or settings.TOP_K
    matches = []
    qvec = embed_text(query)
    for r in search_by_vector(db, qvec, top_k=top_k):
        matches.append({
            "id": str(r.get("id")), "video_id": str(r.get("video_id")), "filename": r.get("filename"),
            "start_sec": float(r.get("start_sec")), "end_sec": float(r.get("end_sec")),
            "summary": r.get("summary"), "title": r.get("title")
        })
    if not matches:
        return {"answer":"I couldn't find any highlights matching your question in the database.","matches":[]}
    matches.sort(key=lambda m: (m["filename"], m["start_sec"]))
    bullets = []
    for m in matches:
        span = f"{sec_to_timestamp(m['start_sec'])}-{sec_to_timestamp(m['end_sec'])}"
        txt = m['summary'] or m['title'] or "(no summary)"
        bullets.append(f"• {m['filename']} [{span}]: {txt}")
    return {"answer":"", "matches": matches}
