from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from app.config import get_settings

settings = get_settings()

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

engine = create_engine(SQLALCHEMY_DATABASE_URL)

Base = declarative_base()
