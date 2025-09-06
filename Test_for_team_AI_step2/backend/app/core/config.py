import os
class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL","postgresql+psycopg2://postgres:postgres@db:5432/videos")
    SEARCH_MODE = os.getenv("SEARCH_MODE","hybrid")  # vector|keyword|hybrid
    TOP_K = int(os.getenv("TOP_K","5"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
settings = Settings()
