from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional

from app.core.database import Base

class User(Base):
    """User model for authentication and profile management"""
    
    __tablename__ = "users"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Personal information (optional for privacy)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    
    # Mental health profile
    mental_health_history = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    emergency_contact = Column(String(255), nullable=True)
    
    # Privacy settings
    data_sharing_consent = Column(Boolean, default=False)
    research_participation = Column(Boolean, default=False)
    
    # Tracking
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Risk assessment
    current_risk_level = Column(String(20), default="low")  # low, medium, high, critical
    last_assessment_date = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
    
    @property
    def full_name(self) -> Optional[str]:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name
    
    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding sensitive data)"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "current_risk_level": self.current_risk_level,
            "created_at": self.created_at,
            "last_login": self.last_login
        }