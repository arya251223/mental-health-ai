from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger

from app.core.database import get_db
from app.models.user import User
from app.models.schemas import (
    MentalHealthAssessment,
    AssessmentResult,
    APIResponse,
    CrisisAssessment,
    RiskLevel
)
from app.api.routes.auth import get_current_user
from app.services.mental_health_service import mental_health_analyzer
from app.api.routes.users import UserService

router = APIRouter()

class MentalHealthService:
    """Mental health assessment and management service"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def process_assessment(
        self, 
        user: User, 
        assessment: MentalHealthAssessment
    ) -> Dict[str, Any]:
        """Process mental health assessment"""
        try:
            # Analyze the text input
            result = await mental_health_analyzer.analyze_text(assessment)
            
            # Update user's risk level if changed
            if result.risk_level.value != user.current_risk_level:
                user_service = UserService(self.db)
                user_service.update_risk_level(user.id, result.risk_level)
            
            # Log assessment for monitoring
            logger.info(f"Assessment completed for user {user.id}: {result.risk_level.value}")
            
            # Check for crisis situation
            crisis_detected = result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            response_data = {
                "assessment_result": result,
                "user_updated": True,
                "crisis_detected": crisis_detected,
                "timestamp": datetime.utcnow()
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing assessment: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Assessment processing failed"
            )
    
    def get_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """Get personalized recommendations based on risk level"""
        recommendations = {
            RiskLevel.LOW: [
                "Continue practicing self-care",
                "Maintain regular exercise routine",
                "Stay connected with friends and family",
                "Practice mindfulness or meditation"
            ],
            RiskLevel.MEDIUM: [
                "Consider speaking with a counselor",
                "Practice stress management techniques",
                "Ensure adequate sleep (7-9 hours)",
                "Limit alcohol and caffeine intake",
                "Join a support group"
            ],
            RiskLevel.HIGH: [
                "Schedule appointment with mental health professional",
                "Contact your doctor",
                "Reach out to trusted friends or family",
                "Use crisis helpline if needed: 988",
                "Avoid making major life decisions"
            ],
            RiskLevel.CRITICAL: [
                "Seek immediate professional help",
                "Contact emergency services if in immediate danger",
                "Call crisis helpline: 988",
                "Do not be alone - contact someone immediately",
                "Go to nearest emergency room if necessary"
            ]
        }
        
        return recommendations.get(risk_level, recommendations[RiskLevel.LOW])

async def handle_crisis_situation(user: User, assessment_result: AssessmentResult):
    """Background task to handle crisis situations"""
    try:
        if assessment_result.risk_level == RiskLevel.CRITICAL:
            # In a real system, this would trigger:
            # - Immediate notification to emergency contacts
            # - Alert to mental health professionals
            # - SMS/email with crisis resources
            logger.critical(f"CRISIS DETECTED for user {user.id}: {user.email}")
            
        elif assessment_result.risk_level == RiskLevel.HIGH:
            # Send supportive resources and encourage professional help
            logger.warning(f"HIGH RISK detected for user {user.id}: {user.email}")
            
    except Exception as e:
        logger.error(f"Error handling crisis situation: {str(e)}")

@router.post("/assess", response_model=APIResponse)
async def mental_health_assessment(
    assessment: MentalHealthAssessment,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform mental health assessment"""
    try:
        mental_health_service = MentalHealthService(db)
        result_data = await mental_health_service.process_assessment(
            current_user, 
            assessment
        )
        
        assessment_result = result_data["assessment_result"]
        
        # Handle crisis situations in background
        if result_data["crisis_detected"]:
            background_tasks.add_task(
                handle_crisis_situation, 
                current_user, 
                assessment_result
            )
        
        return APIResponse(
            success=True,
            message="Assessment completed successfully",
            data={
                "risk_level": assessment_result.risk_level.value,
                "predicted_conditions": [c.value for c in assessment_result.predicted_conditions],
                "confidence_scores": assessment_result.confidence_scores,
                "recommendations": assessment_result.recommendations,
                "crisis_indicators": assessment_result.crisis_indicators,
                "requires_immediate_attention": result_data["crisis_detected"],
                "assessment_id": f"assess_{current_user.id}_{int(datetime.utcnow().timestamp())}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assessment error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment failed"
        )

@router.get("/recommendations", response_model=APIResponse)
async def get_personalized_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized recommendations based on user's current risk level"""
    try:
        mental_health_service = MentalHealthService(db)
        
        # Convert string risk level back to enum
        risk_level = RiskLevel(current_user.current_risk_level)
        recommendations = mental_health_service.get_recommendations(risk_level)
        
        return APIResponse(
            success=True,
            message="Recommendations retrieved",
            data={
                "current_risk_level": risk_level.value,
                "recommendations": recommendations,
                "last_assessment": current_user.last_assessment_date,
                "emergency_resources": {
                    "crisis_line": "988 - Suicide & Crisis Lifeline",
                    "text_line": "Text HOME to 741741",
                    "emergency": "911 for immediate emergencies"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recommendations"
        )

@router.get("/resources", response_model=APIResponse)
async def get_mental_health_resources():
    """Get mental health resources and contact information"""
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
