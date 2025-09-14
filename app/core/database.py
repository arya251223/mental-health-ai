from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os

from app.core.config import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base model class
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()