from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base

class FamilyMember(Base):
    __tablename__ = "family_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    relation: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    samples = relationship("VoiceSample", back_populates="family_member", cascade="all, delete-orphan")

class VoiceSample(Base):
    __tablename__ = "voice_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    family_member_id: Mapped[int] = mapped_column(ForeignKey("family_members.id"))
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    family_member = relationship("FamilyMember", back_populates="samples")

class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int | None] = mapped_column(ForeignKey("voice_samples.id"), nullable=True)
    target_member_id: Mapped[int | None] = mapped_column(ForeignKey("family_members.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(40), default="combined")
    aasist_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ecapa_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    whisper_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(120), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
