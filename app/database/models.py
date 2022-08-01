from sqlalchemy import Column, Float, ForeignKey, Integer, String
from app.database.base import Base, engine
from sqlalchemy.orm import relationship


class MediaFile(Base):
    __tablename__ = "media_files"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), index=True)
    predictions = relationship('Prediction', backref="media_file")


class Prediction(Base):
    __tablename__ = "confidence_score"
    # Sql alchemy model does require primary key
    id = Column(Integer, primary_key=True, index=True)
    media_id = Column(Integer, ForeignKey("media_files.id"))
    phrase = Column(String(200))
    time = Column(Integer)
    confidence = Column(Float)


Base.metadata.create_all(engine)
