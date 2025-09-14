from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from app.core.database import get_db
from app.core.security import (
    create_access_token, 
    hash_password, 
    verify_password,
    verify_token
)
from app.models.user import User
from app.models.schemas import (
    UserCreate, 
    UserLogin, 
    UserResponse, 
    Token, 
    APIResponse
)

router = APIRouter()
security = HTTPBearer()

class AuthService:
    """Authentication service with comprehensive user management"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create new user account"""
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(
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
            
            self.db.add(new_user)
            self.db.commit()
            self.db.refresh(new_user)
            
            logger.info(f"New user created: {new_user.email}")
            return new_user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user account"
            )
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user login"""
        try:
            user = self.db.query(User).filter(User.email == email).first()
            
            if not user:
                logger.warning(f"Login attempt with non-existent email: {email}")
                return None
            
            if not user.is_active:
                logger.warning(f"Login attempt with inactive account: {email}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Account is deactivated"
                )
            
            if not verify_password(password, user.hashed_password):
                logger.warning(f"Invalid password attempt for: {email}")
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Successful login: {email}")
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    def get_user_by_token(self, token: str) -> User:
        """Get user from JWT token"""
        try:
            payload = verify_token(token)
            user_id = payload.get("user_id")
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            user = self.db.query(User).filter(User.id == user_id).first()
            
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

# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    auth_service = AuthService(db)
    return auth_service.get_user_by_token(credentials.credentials)

@router.post("/register", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user account"""
    try:
        auth_service = AuthService(db)
        new_user = auth_service.create_user(user_data)
        
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
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=Token)
async def login_user(
    login_data: UserLogin,
    db: Session = Depends(get_db)
):
    """User login endpoint"""
    try:
        auth_service = AuthService(db)
        user = auth_service.authenticate_user(login_data.email, login_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create access token
        access_token = create_access_token(
            data={
                "user_id": user.id,
                "email": user.email,
                "username": user.username
            }
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=7 * 24 * 60 * 60  # 7 days in seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return UserResponse.from_orm(current_user)

@router.post("/logout", response_model=APIResponse)
async def logout_user(
    current_user: User = Depends(get_current_user)
):
    """User logout endpoint"""
    # In a production system, you might want to blacklist the token
    # For now, we'll just return a success message
    logger.info(f"User logged out: {current_user.email}")
    
    return APIResponse(
        success=True,
        message="Successfully logged out"
    )

@router.post("/refresh-token", response_model=Token)
async def refresh_access_token(
    current_user: User = Depends(get_current_user)
):
    """Refresh access token"""
    try:
        # Create new access token
        access_token = create_access_token(
            data={
                "user_id": current_user.id,
                "email": current_user.email,
                "username": current_user.username
            }
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=7 * 24 * 60 * 60
        )
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )
