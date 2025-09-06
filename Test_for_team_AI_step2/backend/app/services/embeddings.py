from sentence_transformers import SentenceTransformer
from ..core.config import settings

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def embed_text(text: str) -> list[float]:
    vec = get_embedder().encode(text)
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)
