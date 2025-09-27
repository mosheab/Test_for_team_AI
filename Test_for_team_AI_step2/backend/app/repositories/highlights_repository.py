from sqlalchemy import select, text
from sqlalchemy.orm import Session
from ..db.models import Highlight, Video

def search_by_keywords(db: Session, query: str, top_k: int = 5):
    q = f"%{query.lower()}%"
    stmt = (select(Highlight, Video)
            .join(Video, Highlight.video_id==Video.id)
            .where((Highlight.summary.ilike(q)) | (Highlight.title.ilike(q)))
            .order_by(Highlight.start_sec.asc())
            .limit(top_k))
    return list(db.execute(stmt).all())

def search_by_vector(db: Session, query_vec: list[float], top_k: int = 5):
    stmt = text("""
        SELECT h.*, v.filename
        FROM highlights AS h
        JOIN videos AS v ON v.id = h.video_id
        ORDER BY h.embedding <-> CAST(:qvec AS vector)
        LIMIT :k
    """)
    return db.execute(stmt, {"qvec": query_vec, "k": top_k}).mappings().all()