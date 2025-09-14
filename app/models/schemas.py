from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Enums for type safety
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class MentalHealthCondition(str, Enum):
    DEPRESSION = "depression"
    ANXIETY = "anxiety" 
    BIPOLAR = "bipolar"
    ADHD = "adhd"
    OCD = "ocd"
    PTSD = "ptsd"

class MessageType(str, Enum):
    USER = "user"
    AI = "ai"
    SYSTEM = "system"

# User Schemas
class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserCreate(UserBase):
    """Schema for user creation"""
    password: str
    age: Optional[int] = None
    data_sharing_consent: bool = False
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserUpdate(BaseModel):
    """Schema for user updates"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    mental_health_history: Optional[str] = None
    current_medications: Optional[str] = None
    emergency_contact: Optional[str] = None

class UserResponse(UserBase):
    """Schema for user response"""
    id: int
    is_active: bool
    is_verified: bool
    current_risk_level: RiskLevel
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Authentication Schemas
class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str

class Token(BaseModel):
    """Schema for authentication token"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    """Schema for token data"""
    user_id: Optional[int] = None
    email: Optional[str] = None

# Mental Health Assessment Schemas
class MentalHealthAssessment(BaseModel):
    """Schema for mental health assessment"""
    text_input: str
    assessment_type: str = "general"
    previous_context: Optional[List[str]] = None
    
    @validator('text_input')
    def validate_text_input(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Text input must be at least 10 characters long')
        return v.strip()

class AssessmentResult(BaseModel):
    """Schema for assessment results"""
    risk_level: RiskLevel
    predicted_conditions: List[MentalHealthCondition]
    confidence_scores: Dict[str, float]
    recommendations: List[str]
    crisis_indicators: List[str]
    timestamp: datetime

# Chat Schemas
class ChatMessage(BaseModel):
    """Schema for chat messages"""
    content: str
    message_type: MessageType = MessageType.USER
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Schema for AI chat responses"""
    message: str
    assessment: Optional[AssessmentResult] = None
    suggested_resources: Optional[List[str]] = None
    requires_human_intervention: bool = False

class ConversationHistory(BaseModel):
    """Schema for conversation history"""
    messages: List[ChatMessage]
    summary: Optional[str] = None
    total_messages: int
    conversation_start: datetime

# Crisis Detection Schemas
class CrisisAssessment(BaseModel):
    """Schema for crisis assessment"""
    is_crisis: bool
    severity_level: int  # 1-10 scale
    crisis_indicators: List[str]
    immediate_actions: List[str]
    professional_referral_needed: bool
    emergency_contact_needed: bool

# API Response Schemas
class APIResponse(BaseModel):
    """Base API response schema"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    service: str
    version: str
    timestamp: datetime = datetime.utcnow()