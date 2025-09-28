import math
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

from sqlalchemy.orm import Session


def search_by_vector(db: Session, query_vec: list[float], top_k: int = 10, max_dist: float = 1.2):
    if not query_vec:
        return []
    stmt = (
        text("""
            SELECT
              h.id,
              h.video_id,
              h.start_sec,
              h.end_sec,
              h.title,
              h.summary,
              v.filename,
              (h.embedding <-> :qvec) AS dist
            FROM highlights AS h
            JOIN videos     AS v ON v.id = h.video_id
            WHERE (h.embedding <-> :qvec) <= :max_dist
            ORDER BY dist ASC
            LIMIT :k
        """)
        .bindparams(
            bindparam("qvec", query_vec, type_=Vector(384)),
            bindparam("max_dist", float(max_dist)),
            bindparam("k", int(top_k)),
        )
    )
    return db.execute(stmt).mappings().all()