# =============================================================================
# FILE: app/services/integrated_mental_health.py - Updated Integration Service
# =============================================================================

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from app.models.schemas import (
    MentalHealthAssessment,
    AssessmentResult,
    RiskLevel,
    MentalHealthCondition,
    CrisisAssessment,
    ChatMessage,
    ChatResponse
)
from app.services.ai_models import ai_models
from app.services.advanced_mental_health import advanced_analyzer
from app.services.conversation_ai import conversation_ai
from app.services.model_inference import model_inference

class IntegratedMentalHealthService:
    """Integrated service combining all mental health AI capabilities"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize all AI services"""
        try:
            logger.info("🧠 Initializing Integrated Mental Health AI Service...")
            
            # Initialize AI models
            await ai_models.initialize_models()
            
            # Load any custom trained models
            self._load_custom_models()
            
            self.initialized = True
            logger.info("✅ Integrated Mental Health Service ready!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize service: {str(e)}")
            raise
    
    def _load_custom_models(self):
        """Load any custom trained models"""
        try:
            # Attempt to load custom models if they exist
            model_types = ["risk_assessment", "crisis_detection", "condition_classification"]
            
            for model_type in model_types:
                success = model_inference.load_model(model_type)
                if success:
                    logger.info(f"✅ Loaded custom {model_type} model")
                else:
                    logger.info(f"ℹ️  Using pre-trained models for {model_type}")
                    
        except Exception as e:
            logger.warning(f"Custom model loading failed: {str(e)}")
    
    async def comprehensive_assessment(
        self,
        assessment: MentalHealthAssessment,
        user_history: Optional[List[str]] = None
    ) -> AssessmentResult:
        """Perform comprehensive mental health assessment"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use advanced analyzer for comprehensive analysis
            result = await advanced_analyzer.comprehensive_analysis(
                assessment, user_history
            )
            
            # Enhance with custom model predictions if available
            text = assessment.text_input
            
            # Add custom risk prediction
            risk_prediction = model_inference.predict_risk_level(text)
            if risk_prediction.get("method") != "fallback_heuristic":
                # Use custom model result if available
                custom_risk = risk_prediction.get("predicted_risk", "low")
                if custom_risk in ["low", "medium", "high", "critical"]:
                    result.risk_level = RiskLevel(custom_risk)
            
            # Add custom condition predictions
            condition_predictions = model_inference.predict_conditions(text)
            for condition, score in condition_predictions.items():
                if score > 0.3:  # Threshold for inclusion
                    try:
                        condition_enum = MentalHealthCondition(condition)
                        if condition_enum not in result.predicted_conditions:
                            result.predicted_conditions.append(condition_enum)
                        result.confidence_scores[condition_enum] = score
                    except ValueError:
                        # Skip unknown conditions
                        pass
            
            logger.info(f"Assessment completed: {result.risk_level.value} risk level")
            return result
            
        except Exception as e:
            logger.error(f"Assessment failed: {str(e)}")
            raise
    
    async def crisis_intervention(self, text: str, user_id: str) -> Dict[str, Any]:
        """Handle crisis situations with immediate intervention"""
        
        try:
            # Quick crisis detection
            crisis_result = model_inference.predict_crisis(text)
            
            if crisis_result["is_crisis"]:
                logger.critical(f"CRISIS DETECTED for user {user_id}: severity {crisis_result['severity_level']}")
                
                # Generate immediate crisis response
                crisis_response = {
                    "is_crisis": True,
                    "severity": crisis_result["severity_level"],
                    "immediate_actions": [
                        "🚨 Contact 988 (Suicide & Crisis Lifeline) immediately",
                        "📱 Text HOME to 741741 (Crisis Text Line)",
                        "🚑 Call 911 if you're in immediate physical danger",
                        "👥 Reach out to a trusted person right now"
                    ],
                    "professional_resources": [
                        "National Suicide Prevention Lifeline: 988",
                        "Crisis Text Line: Text HOME to 741741",
                        "SAMHSA National Helpline: 1-800-662-4357",
                        "Emergency Services: 911"
                    ],
                    "safety_planning": [
                        "Remove any means of self-harm from your immediate area",
                        "Stay with someone or go to a safe place",
                        "Create a list of people you can contact",
                        "Consider going to the nearest emergency room"
                    ],
                    "follow_up": "immediate_professional_intervention_required"
                }
                
                return crisis_response
            
            return {"is_crisis": False}
            
        except Exception as e:
            logger.error(f"Crisis intervention failed: {str(e)}")
            # Always err on the side of caution
            return {
                "is_crisis": True,
                "severity": 5,
                "immediate_actions": ["Contact 988 immediately for safety"],
                "error": "Assessment failed - please contact crisis services"
            }
    
    async def intelligent_chat_response(
        self,
        message: str,
        user_id: str,
        conversation_history: List[ChatMessage]
    ) -> ChatResponse:
        """Generate intelligent chat response with crisis monitoring"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            # Check for crisis situation first
            crisis_check = await self.crisis_intervention(message, user_id)
            
            if crisis_check.get("is_crisis"):
                # Return crisis intervention response
                return ChatResponse(
                    message="I'm very concerned about what you've shared. Your safety is the most important thing right now.",
                    requires_human_intervention=True,
                    suggested_resources=crisis_check.get("immediate_actions", []),
                    assessment=None
                )
            
            # Generate normal conversational response
            response = await conversation_ai.generate_response(
                message, user_id, conversation_history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Chat response generation failed: {str(e)}")
            return ChatResponse(
                message="I'm here to listen and support you. Could you tell me more about how you're feeling?",
                suggested_resources=[
                    "If you're experiencing a crisis, please contact 988 immediately",
                    "Your mental health matters and support is available"
                ]
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and capabilities"""
        
        return {
            "service_name": "Integrated Mental Health AI",
            "version": "1.0.0",
            "initialized": self.initialized,
            "capabilities": {
                "comprehensive_assessment": True,
                "crisis_detection": True,
                "conversational_ai": True,
                "risk_evaluation": True,
                "condition_classification": True,
                "therapeutic_responses": True
            },
            "ai_models": {
                "sentiment_analysis": ai_models.pipelines.get("sentiment") is not None,
                "emotion_detection": ai_models.pipelines.get("emotion") is not None,
                "mental_health_classification": ai_models.pipelines.get("mental_health") is not None,
                "crisis_detection": ai_models.pipelines.get("crisis") is not None,
                "embeddings": ai_models.embeddings_model is not None
            },
            "custom_models": {
                "risk_assessment": "risk_assessment" in model_inference.loaded_models,
                "crisis_detection": "crisis_detection" in model_inference.loaded_models,
                "condition_classification": "condition_classification" in model_inference.loaded_models
            },
            "status": "operational" if self.initialized else "initializing",
            "timestamp": datetime.now().isoformat()
        }

# Global integrated service instance
integrated_service = IntegratedMentalHealthService()
