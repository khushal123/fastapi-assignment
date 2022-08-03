from typing import Any, List
from sqlalchemy.orm import joinedload, contains_eager
from app.database.confidence.models import Prediction, MediaFile
import logging
logger = logging.getLogger("app")


class AppService:
    def __init__(self, db) -> None:
        self.db = db

    def get_media_by_id(id: int):
        pass

    def get_media_by_name(self, file_name: str, duration: int):
        """ 
        Get or create media file object from db. 

        Args:
            file_name: file path of audio file
        """
        media_file = None
        media_file = self.db.query(MediaFile).filter(
            MediaFile.file_name == file_name).first()
        if media_file is None:
            media_file = MediaFile(file_name=file_name, duration=duration)
            self.db.add(media_file)
            self.db.commit()
            self.db.flush()
            self.db.refresh(media_file)
        return media_file

    def save_bulk(self, predictions: List[Prediction]):
        self.db.bulk_save_objects(predictions)
        self.db.commit()
        return True

    def get_confidence_list(self, page: int, limit: int) -> Any:
        try:
            # this is for limiting the number of resutls
            confidence_list = self.db.query(MediaFile).join(Prediction).options(contains_eager(
                MediaFile.confidences
            )).order_by(Prediction.id.desc()).limit(limit).all()

            return confidence_list
        except Exception as e:
            logger.error(e)
