from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from loguru import logger

from app.core.database import get_db
from app.models.user import User
from app.models.schemas import (
    UserResponse, 
    UserUpdate, 
    APIResponse,
    RiskLevel
)
from app.api.routes.auth import get_current_user

router = APIRouter()

class UserService:
    """User management service"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def update_user_profile(self, user_id: int, update_data: UserUpdate) -> User:
        """Update user profile information"""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Update fields if provided
            update_fields = update_data.dict(exclude_unset=True)
            
            for field, value in update_fields.items():
                if hasattr(user, field):
                    setattr(user, field, value)
            
            user.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"User profile updated: {user.email}")
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile"
            )
    
    def update_risk_level(self, user_id: int, risk_level: RiskLevel) -> User:
        """Update user's mental health risk level"""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            user.current_risk_level = risk_level.value
            user.last_assessment_date = datetime.utcnow()
            user.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"Risk level updated for {user.email}: {risk_level.value}")
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating risk level: {str(e)}")
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update risk level"
            )
    
    def deactivate_user(self, user_id: int) -> User:
        """Deactivate user account"""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            user.is_active = False
            user.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"User account deactivated: {user.email}")
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deactivating user: {str(e)}")
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to deactivate account"
            )

@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get user profile information"""
    return UserResponse.from_orm(current_user)

@router.put("/profile", response_model=APIResponse)
async def update_user_profile(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    try:
        user_service = UserService(db)
        updated_user = user_service.update_user_profile(current_user.id, update_data)
        
        return APIResponse(
            success=True,
            message="Profile updated successfully",
            data=UserResponse.from_orm(updated_user).dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.delete("/account", response_model=APIResponse)
async def deactivate_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deactivate user account"""
    try:
        user_service = UserService(db)
        user_service.deactivate_user(current_user.id)
        
        return APIResponse(
            success=True,
            message="Account deactivated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deactivation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate account"
        )

@router.get("/stats", response_model=APIResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user statistics and mental health metrics"""
    try:
        # Calculate user statistics
        account_age = (datetime.utcnow() - current_user.created_at).days
        
        # In a real application, you'd query actual usage data
        # For now, we'll return mock statistics
        stats = {
            "account_age_days": account_age,
            "current_risk_level": current_user.current_risk_level,
            "last_assessment": current_user.last_assessment_date,
            "last_login": current_user.last_login,
            "data_sharing_consent": current_user.data_sharing_consent,
            "total_conversations": 0,  # Would be calculated from chat history
            "assessments_completed": 0,  # Would be calculated from assessments
            "improvement_score": 0.0  # Would be calculated from progress tracking
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