from functools import lru_cache
import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", None)
    MODEL_CONFIDENCE_THRESHOLD = 0.9
    SAMPLE_RATE = 8000


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
