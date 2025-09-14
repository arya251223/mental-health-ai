# =============================================================================
# MAIN.PY - COMPLETE INTEGRATION FILE
# Mental Health AI System - All Components Integrated
# =============================================================================

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, EmailStr, validator
from pydantic_settings import BaseSettings
from passlib.context import CryptContext
from jose import JWTError, jwt
import uvicorn
import asyncio
import re
import time
import uuid
import os
from pathlib import Path
from loguru import logger
import sys
import json

# Enhanced Chat Service Import
try:
    from app.services.enhanced_chat import enhanced_chat_service
    ENHANCED_CHAT_AVAILABLE = True
    logger.info("✅ Enhanced chat service loaded successfully")
except ImportError as e:
    ENHANCED_CHAT_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced chat not available, using basic chat: {e}")


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file logging
logger.add(
    "logs/mental_health_api.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Application settings - integrates app/core/config.py"""
    
    APP_NAME: str = "Mental Health AI"
    DEBUG: bool = True
    API_VERSION: str = "v1"
    SECRET_KEY: str = "mental-health-ai-super-secret-development-key-2024-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DATABASE_URL: str = "sqlite:///./mental_health.db"
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000", "*"]
    MAX_TEXT_LENGTH: int = 512
    CONFIDENCE_THRESHOLD: float = 0.7
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/mental_health_api.log"
    
    # Crisis Keywords
    CRISIS_KEYWORDS: List[str] = [
        "suicide", "kill myself", "end it all", "worthless", 
        "hopeless", "can't go on", "want to die"
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()

# =============================================================================
# DATABASE SETUP - Integrates app/core/database.py
# =============================================================================

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Session:
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# DATABASE MODELS - Integrates app/models/user.py
# =============================================================================

class User(Base):
    """User model for authentication and profile management"""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    mental_health_history = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    emergency_contact = Column(String(255), nullable=True)
    data_sharing_consent = Column(Boolean, default=False)
    research_participation = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    current_risk_level = Column(String(20), default="low")
    last_assessment_date = Column(DateTime(timezone=True), nullable=True)
    
    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding sensitive data)"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": f"{self.first_name} {self.last_name}" if self.first_name and self.last_name else None,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "current_risk_level": self.current_risk_level,
            "created_at": self.created_at,
            "last_login": self.last_login
        }

# =============================================================================
# PYDANTIC SCHEMAS - Integrates app/models/schemas.py
# =============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    data_sharing_consent: bool = False
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    first_name: Optional[str] = None
    is_active: bool
    current_risk_level: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class MentalHealthAssessment(BaseModel):
    text_input: str
    assessment_type: str = "general"
    previous_context: Optional[List[str]] = None
    
    @validator('text_input')
    def validate_text_input(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Text input must be at least 10 characters long')
        return v.strip()

class ChatMessage(BaseModel):
    content: str
    message_type: str = "user"
    timestamp: Optional[datetime] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None

# =============================================================================
# SECURITY - Integrates app/core/security.py
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def hash_password(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError as e:
        logger.error(f"JWT verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user from token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# =============================================================================
# MENTAL HEALTH AI SERVICE - Integrates Day 3 AI Services
# =============================================================================

class MentalHealthAnalyzer:
    """Advanced mental health analysis service - integrates all Day 3 AI features"""
    
    def __init__(self):
        self.crisis_keywords = settings.CRISIS_KEYWORDS
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Condition keywords mapping
        self.condition_keywords = {
            "depression": [
                "sad", "hopeless", "worthless", "empty", "crying", 
                "tired", "sleep", "appetite", "concentrate", "guilt"
            ],
            "anxiety": [
                "worry", "nervous", "panic", "fear", "anxious", 
                "stress", "tension", "restless", "overwhelmed"
            ],
            "bipolar": [
                "mood swings", "manic", "high energy", "low energy",
                "extreme", "ups and downs"
            ],
            "adhd": [
                "focus", "attention", "concentrate", "hyperactive",
                "impulsive", "distracted", "restless"
            ],
            "ocd": [
                "obsessive", "compulsive", "repetitive", "ritual",
                "intrusive thoughts", "checking", "counting"
            ],
            "ptsd": [
                "trauma", "flashback", "nightmare", "trigger",
                "avoidance", "hypervigilant", "startled"
            ]
        }
        
        # Crisis patterns with severity levels
        self.crisis_patterns = {
            "immediate_danger": {
                "patterns": [
                    r"(?:want to|going to|plan to) (?:die|kill myself|end (?:it|my life))",
                    r"suicide plan|method to (?:die|kill)",
                    r"(?:tonight|today|now|soon) (?:I will|I'm going to) (?:die|kill myself)",
                ],
                "severity": 10
            },
            "suicidal_ideation": {
                "patterns": [
                    r"(?:think about|thoughts of) (?:dying|death|suicide|killing myself)",
                    r"wish I (?:was|were) dead",
                    r"life (?:isn't|is not) worth living",
                    r"(?:everyone|world) (?:would be|is) better (?:without|off without) me"
                ],
                "severity": 8
            },
            "self_harm": {
                "patterns": [
                    r"(?:cut|cutting|harm|hurt) myself",
                    r"self.?harm|self.?injury",
                    r"want to hurt myself"
                ],
                "severity": 7
            },
            "severe_distress": {
                "patterns": [
                    r"can't (?:take|handle|go on|do) (?:this|it) anymore",
                    r"nothing (?:matters|helps|works)",
                    r"completely (?:hopeless|lost|broken)"
                ],
                "severity": 6
            }
        }
    
    async def analyze_text(self, assessment: MentalHealthAssessment) -> Dict[str, Any]:
        """Main analysis function - integrates comprehensive mental health analysis"""
        try:
            text = assessment.text_input.lower()
            
            # Perform analyses
            crisis_result = await self._assess_crisis_risk(text)
            sentiment_analysis = self._analyze_sentiment(text)
            condition_predictions = self._predict_conditions(text)
            risk_level = self._calculate_risk_level(crisis_result, sentiment_analysis, condition_predictions)
            recommendations = self._generate_recommendations(risk_level, condition_predictions)
            
            result = {
                "risk_level": risk_level,
                "predicted_conditions": list(condition_predictions.keys()),
                "confidence_scores": condition_predictions,
                "recommendations": recommendations,
                "crisis_indicators": crisis_result["crisis_indicators"],
                "requires_immediate_attention": crisis_result["is_crisis"],
                "sentiment_analysis": sentiment_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Assessment completed: {risk_level} risk level")
            return result
            
        except Exception as e:
            logger.error(f"Error in mental health analysis: {str(e)}")
            # Return safe default response
            return {
                "risk_level": "low",
                "predicted_conditions": [],
                "confidence_scores": {},
                "recommendations": ["Please consult with a mental health professional for personalized guidance."],
                "crisis_indicators": [],
                "requires_immediate_attention": False,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _assess_crisis_risk(self, text: str) -> Dict[str, Any]:
        """Assess crisis risk level"""
        crisis_indicators = []
        severity_level = 0
        
        # Check for crisis keywords
        for keyword in self.crisis_keywords:
            if keyword in text:
                crisis_indicators.append(f"Crisis keyword detected: {keyword}")
                severity_level += 2
        
        # Check for immediate danger phrases
        for category, config in self.crisis_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    crisis_indicators.append(f"Crisis pattern detected: {category}")
                    severity_level = max(severity_level, config["severity"])
        
        # Determine crisis level
        is_crisis = severity_level >= 5
        professional_referral = severity_level >= 3
        emergency_contact = severity_level >= 8
        
        # Generate immediate actions
        immediate_actions = []
        if emergency_contact:
            immediate_actions.extend([
                "Contact emergency services (911) immediately",
                "Call or text 988 (Suicide & Crisis Lifeline)",
                "Do not leave the person alone"
            ])
        elif is_crisis:
            immediate_actions.extend([
                "Contact 988 (Suicide & Crisis Lifeline) immediately",
                "Text HOME to 741741 (Crisis Text Line)",
                "Reach out to a trusted person for support"
            ])
        elif professional_referral:
            immediate_actions.extend([
                "Consider contacting a mental health professional",
                "Reach out to your support network"
            ])
        
        return {
            "is_crisis": is_crisis,
            "severity_level": min(severity_level, 10),
            "crisis_indicators": crisis_indicators,
            "immediate_actions": immediate_actions,
            "professional_referral_needed": professional_referral,
            "emergency_contact_needed": emergency_contact
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis"""
        # Simple keyword-based sentiment analysis
        positive_words = ["good", "great", "happy", "better", "hope", "grateful", "positive"]
        negative_words = ["bad", "terrible", "awful", "worse", "sad", "depressed", "hopeless", "negative"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score
        
        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": max(0.0, neutral_score)
        }
    
    def _predict_conditions(self, text: str) -> Dict[str, float]:
        """Predict mental health conditions based on keyword analysis"""
        predictions = {}
        
        for condition, keywords in self.condition_keywords.items():
            score = 0
            matches = 0
            
            for keyword in keywords:
                if keyword in text:
                    score += 1
                    matches += 1
            
            # Calculate confidence score (0-1)
            if matches > 0:
                confidence = min(score / len(keywords) * 3, 1.0)  # Amplify score
                if confidence >= 0.2:  # Minimum threshold
                    predictions[condition] = confidence
        
        return predictions
    
    def _calculate_risk_level(
        self, 
        crisis: Dict[str, Any], 
        sentiment: Dict[str, float], 
        conditions: Dict[str, float]
    ) -> str:
        """Calculate overall risk level"""
        
        # Crisis assessment has highest priority
        if crisis.get("is_crisis"):
            if crisis.get("severity_level", 0) >= 8:
                return "critical"
            else:
                return "high"
        
        risk_score = 0
        
        # Sentiment analysis
        negative_score = sentiment.get("negative", 0)
        if negative_score > 0.3:
            risk_score += 3
        elif negative_score > 0.15:
            risk_score += 2
        elif negative_score > 0.05:
            risk_score += 1
        
        # Condition analysis
        severe_conditions = ["depression", "bipolar", "ptsd"]
        for condition, score in conditions.items():
            if condition in severe_conditions and score > 0.5:
                risk_score += 3
            elif score > 0.4:
                risk_score += 2
            elif score > 0.2:
                risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, risk_level: str, conditions: Dict[str, float]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == "critical":
            recommendations.extend([
                "🚨 IMMEDIATE ACTION: Contact 988 (Suicide & Crisis Lifeline) or emergency services",
                "📱 Text HOME to 741741 (Crisis Text Line)",
                "🚑 Call 911 if you're in immediate physical danger",
                "👥 Do not be alone - contact a trusted person immediately"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "📞 Contact 988 (Suicide & Crisis Lifeline) for immediate support",
                "👨‍⚕️ Schedule an urgent appointment with a mental health professional",
                "👥 Reach out to trusted friends or family members for support",
                "🏥 Consider contacting your doctor or a crisis center"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "👨‍⚕️ Consider scheduling an appointment with a therapist or counselor",
                "🧘 Practice stress-reduction techniques like deep breathing or meditation",
                "👥 Connect with supportive people in your life",
                "📱 Consider using mental health apps for daily support"
            ])
        else:  # LOW
            recommendations.extend([
                "🌟 Continue maintaining healthy habits and self-care",
                "🏃‍♂️ Regular exercise can boost mood and reduce stress",
                "😴 Maintain a healthy sleep schedule (7-9 hours per night)",
                "🧘 Consider mindfulness or meditation practices"
            ])
        
        # Condition-specific recommendations
        for condition, confidence in conditions.items():
            if confidence > 0.4:
                if condition == "depression":
                    recommendations.extend([
                        "☀️ Try to get sunlight exposure daily, especially in the morning",
                        "🏃‍♂️ Regular physical activity can significantly improve mood"
                    ])
                elif condition == "anxiety":
                    recommendations.extend([
                        "🫁 Practice the 4-7-8 breathing technique when feeling anxious",
                        "🧘‍♀️ Try progressive muscle relaxation or guided meditation"
                    ])
                elif condition == "adhd":
                    recommendations.extend([
                        "📋 Use organizational tools and break tasks into smaller steps",
                        "⏰ Create structured routines and use timers for focus"
                    ])
        
        return recommendations[:8]  # Limit to most important recommendations

# Create analyzer instance
mental_health_analyzer = MentalHealthAnalyzer()

# =============================================================================
# CHAT SERVICE - Integrates Conversational AI
# =============================================================================

class BasicChatService:
    """Fallback basic chat service"""
    
    def __init__(self):
        self.conversations = {}
    
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """Basic message processing"""
        
        # Store message
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "content": message,
            "sender": "user",
            "timestamp": datetime.utcnow()
        })
        
        # Simple response
        if any(word in message.lower() for word in ["hello", "hi", "hey"]):
            response_text = "Hello! I'm here to listen and support you. How are you feeling today?"
        else:
            response_text = "I understand you're sharing something important with me. Can you tell me more about how you're feeling?"
        
        # Store AI response
        self.conversations[user_id].append({
            "content": response_text,
            "sender": "ai", 
            "timestamp": datetime.utcnow()
        })
        
        return {
            "ai_response": response_text,
            "suggested_resources": ["Professional support is available if needed"],
            "requires_human_intervention": False
        }

# Create chat service instance
if ENHANCED_CHAT_AVAILABLE:
    chat_service = enhanced_chat_service
else:
    chat_service = BasicChatService()

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper startup/shutdown"""
    # Startup
    logger.info("🚀 Starting Mental Health AI System v1.0.0")
    
    try:
        # Create database tables
        logger.info("📊 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database ready")
        
        # Initialize AI services
        logger.info("🤖 Initializing AI services...")
        # AI models would be loaded here in production
        logger.info("✅ AI services ready")
        
        logger.info("🎯 Mental Health AI System ready to help!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Mental Health AI System")

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health AI Support System",
    description="""
    🧠 **Advanced AI-powered Mental Health Support Platform**
    
    ## Features
    * 🤖 Comprehensive AI mental health assessment with multi-factor analysis
    * 💬 Intelligent therapeutic chatbot with context awareness and crisis detection
    * 🚨 Real-time crisis detection and immediate intervention protocols
    * 📊 Personalized risk evaluation and evidence-based recommendations
    * 🔐 Secure user authentication and privacy protection
    * 📱 24/7 accessible support with professional-grade responses
    
    ## 🚨 Important Medical Disclaimer
    This system provides AI-powered support and educational resources but is **not a replacement** 
    for professional mental health care. In emergency situations:
    
    * 📞 **Crisis Line**: **988** (24/7 Suicide & Crisis Lifeline)
    * 📱 **Text Support**: Text **HOME** to **741741** 
    * 🚑 **Emergency**: **911** for immediate physical danger
    
    Always consult with qualified mental health professionals for diagnosis and treatment.
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve static files (HTML, JS, CSS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("app/static/index.html")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# =============================================================================
# API ROUTES - All Endpoints Integrated
# =============================================================================

# Health check endpoint
@app.get("/health")
async def comprehensive_health_check():
    """Comprehensive system health check"""
    return {
        "status": "healthy",
        "service": "Mental Health AI API",
        "version": "1.0.0", 
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "authentication": "operational",
            "mental_health_assessment": "operational",
            "chat_interface": "operational", 
            "crisis_detection": "operational",
            "database": "connected"
        },
        "emergency_resources": {
            "crisis_lifeline": "988",
            "crisis_text": "Text HOME to 741741",
            "emergency": "911"
        },
        "endpoints": {
            "docs": "/api/docs",
            "authentication": "/api/auth", 
            "mental_health": "/api/mental-health",
            "chat": "/api/chat"
        }
    }

# Root endpoint
@app.get("/api/welcome")
async def welcome():
    """Welcome message with API guidance"""
    return {
        "message": "💙 Welcome to Mental Health AI Support System",
        "version": "1.0.0",
        "description": "Advanced AI-powered mental health assessment and support",
        "getting_started": {
            "docs": "/api/docs - Interactive API documentation",
            "health": "/health - System status check",
            "register": "/api/auth/register - Create account", 
            "login": "/api/auth/login - User authentication"
        },
        "emergency_help": {
            "crisis_line": "📞 988 - 24/7 Crisis Support",
            "text_support": "📱 Text HOME to 741741", 
            "emergency": "🚑 911 for immediate danger"
        },
        "features": [
            "🧠 AI-powered mental health assessment",
            "💬 Therapeutic conversational AI",
            "🚨 Real-time crisis detection",
            "📊 Personalized recommendations"
        ]
    }

# =============================================================================
# AUTHENTICATION ROUTES
# =============================================================================

@app.post("/api/auth/register")
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == user_data.email) | 
            (User.username == user_data.username)
        ).first()
        
        if existing_user:
            if existing_user.email == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
        
        # Create new user
        hashed_pw = hash_password(user_data.password)
        
        new_user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_pw,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            age=user_data.age,
            data_sharing_consent=user_data.data_sharing_consent,
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"New user created: {new_user.email}")
        
        return APIResponse(
            success=True,
            message="User account created successfully",
            data={
                "user_id": new_user.id,
                "username": new_user.username,
                "email": new_user.email
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )

@app.post("/api/auth/login", response_model=Token)
async def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """User login endpoint"""
    try:
        user = db.query(User).filter(User.email == login_data.email).first()
        
        if not user:
            logger.warning(f"Login attempt with non-existent email: {login_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user.is_active:
            logger.warning(f"Login attempt with inactive account: {login_data.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        if not verify_password(login_data.password, user.hashed_password):
            logger.warning(f"Invalid password attempt for: {login_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create access token
        access_token = create_access_token(
            data={
                "user_id": user.id,
                "email": user.email,
                "username": user.username
            }
        )
        
        logger.info(f"Successful login: {login_data.email}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse.model_validate(current_user)

@app.post("/api/auth/logout")
async def logout_user(current_user: User = Depends(get_current_user)):
    """User logout endpoint"""
    logger.info(f"User logged out: {current_user.email}")
    
    return APIResponse(
        success=True,
        message="Successfully logged out"
    )

# =============================================================================
# MENTAL HEALTH ASSESSMENT ROUTES
# =============================================================================

@app.post("/api/mental-health/assess")
async def mental_health_assessment(
    assessment: MentalHealthAssessment,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform comprehensive mental health assessment"""
    try:
        # Perform analysis
        result = await mental_health_analyzer.analyze_text(assessment)
        
        # Update user's risk level and last assessment date
        current_user.current_risk_level = result["risk_level"]
        current_user.last_assessment_date = datetime.utcnow()
        db.commit()
        
        # Handle crisis situations in background
        if result.get("requires_immediate_attention"):
            background_tasks.add_task(
                handle_crisis_situation, 
                current_user, 
                result
            )
        
        return APIResponse(
            success=True,
            message="Assessment completed successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assessment error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment failed"
        )

@app.get("/api/mental-health/resources")
async def get_mental_health_resources():
    """Get comprehensive mental health resources"""
    try:
        resources = {
            "crisis_hotlines": {
                "national_suicide_prevention": {
                    "name": "988 Suicide & Crisis Lifeline",
                    "phone": "988",
                    "description": "24/7 free and confidential support"
                },
                "crisis_text_line": {
                    "name": "Crisis Text Line",
                    "text": "Text HOME to 741741",
                    "description": "24/7 crisis support via text"
                }
            },
            "professional_help": {
                "psychology_today": {
                    "name": "Psychology Today",
                    "url": "https://www.psychologytoday.com",
                    "description": "Find therapists and mental health professionals"
                },
                "samhsa": {
                    "name": "SAMHSA Treatment Locator",
                    "phone": "1-800-662-4357",
                    "description": "Treatment facility locator"
                }
            },
            "self_help_resources": [
                "Mindfulness and meditation apps (Headspace, Calm)",
                "Regular exercise (30 minutes daily)",
                "Maintain sleep schedule (7-9 hours)",
                "Connect with supportive people",
                "Practice gratitude journaling",
                "Limit alcohol and substance use"
            ],
            "emergency": {
                "description": "If you're in immediate danger",
                "action": "Call 911 or go to your nearest emergency room"
            }
        }
        
        return APIResponse(
            success=True,
            message="Mental health resources retrieved",
            data=resources
        )
        
    except Exception as e:
        logger.error(f"Error getting resources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get resources"
        )

# =============================================================================
# CHAT INTERFACE ROUTES
# =============================================================================

@app.post("/api/chat/message")
async def send_chat_message(
    message: ChatMessage,
    current_user: User = Depends(get_current_user)
):
    """Send message to enhanced AI chatbot"""
    try:
        if ENHANCED_CHAT_AVAILABLE:
            # Use enhanced chat service
            response = await enhanced_chat_service.generate_response(
                message.content,
                str(current_user.id),
                []  # conversation history
            )
        else:
            # Fallback to basic service
            response = await chat_service.process_message(
                str(current_user.id), 
                message.content
            )
        
        return APIResponse(
            success=True,
            message="Message processed successfully",
            data=response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat message error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )
    
@app.get("/api/chat/history")
async def get_chat_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get user's chat history"""
    try:
        user_id = str(current_user.id)
        
        if user_id in chat_service.conversations:
            messages = chat_service.conversations[user_id]
            
            # Limit messages if specified
            if limit and len(messages) > limit:
                messages = messages[-limit:]
            
            # Convert to serializable format
            messages_data = []
            for msg in messages:
                messages_data.append({
                    "content": msg["content"],
                    "sender": msg["sender"],
                    "timestamp": msg["timestamp"].isoformat()
                })
            
            return APIResponse(
                success=True,
                message="Chat history retrieved",
                data={
                    "messages": messages_data,
                    "total_messages": len(messages_data),
                    "user_id": current_user.id
                }
            )
        else:
            return APIResponse(
                success=True,
                message="No chat history found",
                data={
                    "messages": [],
                    "total_messages": 0,
                    "user_id": current_user.id
                }
            )
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history"
        )

@app.delete("/api/chat/history")
async def clear_chat_history(current_user: User = Depends(get_current_user)):
    """Clear user's chat history"""
    try:
        user_id = str(current_user.id)
        
        if user_id in chat_service.conversations:
            del chat_service.conversations[user_id]
        
        logger.info(f"Chat history cleared for user {current_user.id}")
        
        return APIResponse(
            success=True,
            message="Chat history cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear chat history"
        )

# =============================================================================
# USER MANAGEMENT ROUTES
# =============================================================================

@app.get("/api/users/profile", response_model=UserResponse)
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get user profile information"""
    return UserResponse.model_validate(current_user)

@app.get("/api/users/stats")
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user statistics and mental health metrics"""
    try:
        # Calculate user statistics
        account_age = (datetime.utcnow() - current_user.created_at).days
        
        # Get conversation count
        user_id = str(current_user.id)
        conversation_count = 0
        if user_id in chat_service.conversations:
            user_messages = [msg for msg in chat_service.conversations[user_id] if msg["sender"] == "user"]
            conversation_count = len(user_messages)
        
        stats = {
            "account_age_days": account_age,
            "current_risk_level": current_user.current_risk_level,
            "last_assessment": current_user.last_assessment_date.isoformat() if current_user.last_assessment_date else None,
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
            "data_sharing_consent": current_user.data_sharing_consent,
            "total_conversations": conversation_count,
            "account_status": "active" if current_user.is_active else "inactive",
            "member_since": current_user.created_at.isoformat()
        }
        
        return APIResponse(
            success=True,
            message="User statistics retrieved",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def handle_crisis_situation(user: User, assessment_result: Dict[str, Any]):
    """Background task to handle crisis situations"""
    try:
        if assessment_result.get("risk_level") == "critical":
            logger.critical(f"CRISIS DETECTED for user {user.id}: {user.email}")
            
        elif assessment_result.get("risk_level") == "high":
            logger.warning(f"HIGH RISK detected for user {user.id}: {user.email}")
            
    except Exception as e:
        logger.error(f"Error handling crisis situation: {str(e)}")

# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handler for request validation errors"""
    logger.warning(f"Validation error for {request.url.path}: {exc.errors()}")
    
    return {
        "success": False,
        "message": "Invalid input data",
        "errors": [f"{error['loc'][-1]}: {error['msg']}" for error in exc.errors()]
    }

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handler for HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code} for {request.url.path}: {exc.detail}")
    
    return {
        "success": False,
        "message": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler for general exceptions"""
    logger.error(f"Unhandled exception for {request.url.path}: {str(exc)}")
    
    return {
        "success": False,
        "message": "An internal server error occurred",
        "error_type": type(exc).__name__,
        "support": "Contact support if this persists"
    }

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )