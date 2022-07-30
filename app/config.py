from functools import lru_cache
import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", None)

@lru_cache()
def get_settings():
    return Settings()