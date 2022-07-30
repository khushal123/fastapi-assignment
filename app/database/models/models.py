from sqlalchemy import Column, Float, Integer, String
from database.base import Base
from sqlalchemy.orm import relationship


class MediaFile(Base):
    __tablename__ = "media_files"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255))
    duration = Column(Integer)


class Prediction(Base):
    __tablename__ = "confidence_score"
    media_file = relationship(
        "MediaFile",
        cascade="all,delete-orphan",
        uselist=True
    )
    utterance = Column(String(50))
    time = Column(Integer())
    confidence = Column(Float())
