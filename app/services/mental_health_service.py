from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from textblob import TextBlob
from loguru import logger
import asyncio

from app.models.schemas import (
    MentalHealthAssessment, 
    AssessmentResult, 
    RiskLevel, 
    MentalHealthCondition,
    CrisisAssessment
)
from app.core.config import settings

class MentalHealthAnalyzer:
    """Core mental health analysis service"""
    
    def __init__(self):
        self.crisis_keywords = settings.CRISIS_KEYWORDS
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Condition keywords mapping
        self.condition_keywords = {
            MentalHealthCondition.DEPRESSION: [
                "sad", "hopeless", "worthless", "empty", "crying", 
                "tired", "sleep", "appetite", "concentrate", "guilt"
            ],
            MentalHealthCondition.ANXIETY: [
                "worry", "nervous", "panic", "fear", "anxious", 
                "stress", "tension", "restless", "overwhelmed"
            ],
            MentalHealthCondition.BIPOLAR: [
                "mood swings", "manic", "high energy", "low energy",
                "extreme", "ups and downs"
            ],
            MentalHealthCondition.ADHD: [
                "focus", "attention", "concentrate", "hyperactive",
                "impulsive", "distracted", "restless"
            ],
            MentalHealthCondition.OCD: [
                "obsessive", "compulsive", "repetitive", "ritual",
                "intrusive thoughts", "checking", "counting"
            ],
            MentalHealthCondition.PTSD: [
                "trauma", "flashback", "nightmare", "trigger",
                "avoidance", "hypervigilant", "startled"
            ]
        }
    
    async def analyze_text(self, assessment: MentalHealthAssessment) -> AssessmentResult:
        """Main analysis function"""
        try:
            text = assessment.text_input.lower()
            
            # Perform various analyses
            crisis_result = await self._assess_crisis_risk(text)
            sentiment_analysis = self._analyze_sentiment(text)
            condition_predictions = self._predict_conditions(text)
            risk_level = self._calculate_risk_level(crisis_result, sentiment_analysis, condition_predictions)
            recommendations = self._generate_recommendations(risk_level, condition_predictions)
            
            return AssessmentResult(
                risk_level=risk_level,
                predicted_conditions=list(condition_predictions.keys()),
                confidence_scores=condition_predictions,
                recommendations=recommendations,
                crisis_indicators=crisis_result.crisis_indicators,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in mental health analysis: {str(e)}")
            # Return safe default response
            return AssessmentResult(
                risk_level=RiskLevel.LOW,
                predicted_conditions=[],
                confidence_scores={},
                recommendations=["Please consult with a mental health professional for personalized guidance."],
                crisis_indicators=[],
                timestamp=datetime.utcnow()
            )
    
    async def _assess_crisis_risk(self, text: str) -> CrisisAssessment:
        """Assess crisis risk level"""
        crisis_indicators = []
        severity_level = 0
        
        # Check for crisis keywords
        for keyword in self.crisis_keywords:
            if keyword in text:
                crisis_indicators.append(f"Crisis keyword detected: {keyword}")
                severity_level += 2
        
        # Check for immediate danger phrases
        immediate_danger_patterns = [
            r"want to (die|kill myself)",
            r"going to (hurt|harm) myself",
            r"can't (take|handle) (it|this) anymore",
            r"end it all",
            r"no point in living"
        ]
        
        for pattern in immediate_danger_patterns:
            if re.search(pattern, text):
                crisis_indicators.append(f"Immediate danger language detected")
                severity_level += 3
        
        # Determine crisis level
        is_crisis = severity_level >= 3
        professional_referral = severity_level >= 2
        emergency_contact = severity_level >= 5
        
        # Generate immediate actions based on severity
        immediate_actions = []
        if emergency_contact:
            immediate_actions.extend([
                "Contact emergency services (911) immediately",
                "Do not leave the person alone",
                "Remove any potential means of self-harm"
            ])
        elif is_crisis:
            immediate_actions.extend([
                "Contact a crisis helpline: 988 (Suicide & Crisis Lifeline)",
                "Reach out to a trusted friend or family member",
                "Consider going to the nearest emergency room"
            ])
        elif professional_referral:
            immediate_actions.extend([
                "Schedule an appointment with a mental health professional",
                "Contact your primary care physician",
                "Consider reaching out to a counselor or therapist"
            ])
        
        return CrisisAssessment(
            is_crisis=is_crisis,
            severity_level=min(severity_level, 10),
            crisis_indicators=crisis_indicators,
            immediate_actions=immediate_actions,
            professional_referral_needed=professional_referral,
            emergency_contact_needed=emergency_contact
        )
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        
        return {
            "polarity": blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
            "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        }
    
    def _predict_conditions(self, text: str) -> Dict[MentalHealthCondition, float]:
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
                confidence = min(score / len(keywords), 1.0)
                if confidence >= 0.1:  # Minimum threshold
                    predictions[condition] = confidence
        
        return predictions
    
    def _calculate_risk_level(
        self, 
        crisis: CrisisAssessment, 
        sentiment: Dict[str, float], 
        conditions: Dict[MentalHealthCondition, float]
    ) -> RiskLevel:
        """Calculate overall risk level"""
        
        # Crisis assessment has highest priority
        if crisis.is_crisis:
            if crisis.severity_level >= 7:
                return RiskLevel.CRITICAL
            else:
                return RiskLevel.HIGH
        
        # Sentiment analysis
        polarity = sentiment.get("polarity", 0)
        risk_score = 0
        
        # Negative sentiment increases risk
        if polarity < -0.5:
            risk_score += 3
        elif polarity < -0.2:
            risk_score += 2
        elif polarity < 0:
            risk_score += 1
        
        # Multiple conditions increase risk
        condition_count = len(conditions)
        if condition_count >= 3:
            risk_score += 2
        elif condition_count >= 2:
            risk_score += 1
        
        # High confidence in severe conditions
        severe_conditions = [MentalHealthCondition.DEPRESSION, MentalHealthCondition.BIPOLAR, MentalHealthCondition.PTSD]
        for condition in severe_conditions:
            if condition in conditions and conditions[condition] > 0.6:
                risk_score += 2
        
        # Determine risk level
        if risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(
        self, 
        risk_level: RiskLevel, 
        conditions: Dict[MentalHealthCondition, float]
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # General recommendations based on risk level
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Seek immediate professional help - contact emergency services if necessary",
                "Do not hesitate to reach out to a crisis helpline: 988",
                "Ensure you have someone you can talk to right now"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Consider scheduling an appointment with a mental health professional this week",
                "Reach out to trusted friends or family members for support",
                "Contact a mental health helpline if you need someone to talk to"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Consider speaking with a counselor or therapist",
                "Practice self-care activities that you enjoy",
                "Maintain a regular sleep schedule and healthy diet"
            ])
        else:  # LOW
            recommendations.extend([
                "Continue maintaining healthy habits and self-care",
                "Stay connected with supportive people in your life",
                "Consider mindfulness or relaxation techniques"
            ])
        
        # Condition-specific recommendations
        for condition, confidence in conditions.items():
            if confidence > 0.5:
                if condition == MentalHealthCondition.DEPRESSION:
                    recommendations.extend([
                        "Consider cognitive behavioral therapy (CBT) which is effective for depression",
                        "Regular exercise can help improve mood",
                        "Try to maintain social connections even when it's difficult"
                    ])
                elif condition == MentalHealthCondition.ANXIETY:
                    recommendations.extend([
                        "Practice deep breathing and relaxation techniques",
                        "Consider anxiety management strategies like grounding exercises",
                        "Limit caffeine intake which can worsen anxiety symptoms"
                    ])
                elif condition == MentalHealthCondition.ADHD:
                    recommendations.extend([
                        "Consider speaking with a healthcare provider about ADHD evaluation",
                        "Use organizational tools and techniques to manage daily tasks",
                        "Regular exercise can help improve focus and attention"
                    ])
        
        return recommendations

# Create service instance
mental_health_analyzer = MentalHealthAnalyzer()