from typing import Dict, Any
from sqlalchemy.orm import Session
from ..repositories.highlights_repository import search_by_keywords, search_by_vector
from .embeddings import embed_text
from ..core.config import settings

def sec_to_timestamp(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}" if h else f"{m:02d}:{s:06.3f}"

def answer_query(db: Session, query: str, top_k: int | None = None) -> Dict[str, Any]:
    top_k = top_k or settings.TOP_K
    mode = settings.SEARCH_MODE.lower()
    matches = []
    if mode in ("vector","hybrid"):
        qvec = embed_text(query)
        for r in search_by_vector(db, qvec, top_k=top_k):
            matches.append({
                "id": str(r.get("id")), "video_id": str(r.get("video_id")), "filename": r.get("filename"),
                "start_sec": float(r.get("start_sec")), "end_sec": float(r.get("end_sec")),
                "summary": r.get("summary") or "", "description": r.get("description") or ""
            })
    if mode in ("keyword","hybrid"):
        for h, v in search_by_keywords(db, query, top_k=top_k):
            matches.append({
                "id": str(h.id), "video_id": str(h.video_id), "filename": v.filename,
                "start_sec": float(h.start_sec), "end_sec": float(h.end_sec),
                "summary": h.summary or "", "description": h.description or ""
            })
    if not matches:
        return {"answer":"I couldn’t find any highlights matching your question in the database.","matches":[]}
    seen = set()
    unique_matches = []
    for m in matches:
        key = (m["id"], m["start_sec"], m["end_sec"])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)
    unique_matches.sort(key=lambda m: (m["filename"], m["start_sec"]))
    bullets = []
    for m in unique_matches:
        span = f"{sec_to_timestamp(m['start_sec'])}-{sec_to_timestamp(m['end_sec'])}"
        txt = m['summary'] or m['description'] or "(no summary)"
        bullets.append(f"• {m['filename']} [{span}]: {txt}")
    return {"answer":"", "matches": unique_matches}
