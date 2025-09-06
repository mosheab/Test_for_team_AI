import uuid
from sqlalchemy import Column, String, Float, TIMESTAMP, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from .database import Base

class Video(Base):
    __tablename__ = "videos"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    duration_sec = Column(Float, nullable=True)
    created_at = Column(TIMESTAMP, nullable=True)
    highlights = relationship("Highlight", back_populates="video", cascade="all, delete-orphan")

class Highlight(Base):
    __tablename__ = "highlights"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(TIMESTAMP, nullable=True)
    video = relationship("Video", back_populates="highlights")
