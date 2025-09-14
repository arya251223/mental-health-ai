from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from loguru import logger

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    """Handles all security operations"""
    
    def __init__(self):
        self.algorithm = "HS256"
        self.secret_key = settings.SECRET_KEY
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.error(f"JWT verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

# Create security manager instance
security_manager = SecurityManager()

# Helper functions for backward compatibility
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    return security_manager.create_access_token(data, expires_delta)

def verify_token(token: str) -> Dict[str, Any]:
    return security_manager.verify_token(token)

def hash_password(password: str) -> str:
    return security_manager.hash_password(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return security_manager.verify_password(plain_password, hashed_password)