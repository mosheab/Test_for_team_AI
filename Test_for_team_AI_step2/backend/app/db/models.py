import uuid
from sqlalchemy import Column, String, Float, TIMESTAMP, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Video(Base):
    __tablename__="videos"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    duration_sec = Column(Float)
    created_at = Column(TIMESTAMP)
    highlights = relationship("Highlight", back_populates="video")
    
class Highlight(Base):
    __tablename__="highlights"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)
    description = Column(Text)
    summary = Column(Text)
    embedding = Column(Vector(384))
    created_at = Column(TIMESTAMP)
    video = relationship("Video", back_populates="highlights")
