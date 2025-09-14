from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Basic App Config
    APP_NAME: str = "Mental Health AI"
    DEBUG: bool = True
    API_VERSION: str = "v1"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database
    DATABASE_URL: str = "sqlite:///./mental_health.db"
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # AI/ML Settings
    HUGGING_FACE_API_KEY: str = ""
    MAX_TEXT_LENGTH: int = 512
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Mental Health Crisis Settings
    CRISIS_KEYWORDS: List[str] = [
        "suicide", "kill myself", "end it all", "worthless", 
        "hopeless", "can't go on", "want to die"
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/mental_health_api.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Create logs directory
Path("logs").mkdir(exist_ok=True)