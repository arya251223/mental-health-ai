from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import asyncio
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from textblob import TextBlob
from loguru import logger

# Download required NLTK data (with error handling)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True) 
        nltk.download('vader_lexicon', quiet=True)
        logger.info("📥 Downloaded NLTK data successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to download NLTK data: {str(e)}")

# Import schemas (adjust path as needed)
try:
    from app.models.schemas import (
        MentalHealthAssessment,
        AssessmentResult,
        RiskLevel,
        MentalHealthCondition,
        CrisisAssessment
    )
except ImportError:
    # Fallback definitions if schemas not available
    logger.warning("⚠️ Could not import schemas, using fallback definitions")
    
    class RiskLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class MentalHealthCondition:
        DEPRESSION = "depression"
        ANXIETY = "anxiety"
        BIPOLAR = "bipolar"
        ADHD = "adhd"
        OCD = "ocd"
        PTSD = "ptsd"

class AdvancedMentalHealthAnalyzer:
    """Advanced mental health analysis using multiple AI models and techniques"""
    
    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("✅ Advanced Mental Health Analyzer initialized")
        except Exception as e:
            logger.warning(f"⚠️ NLTK initialization failed: {str(e)}")
            self.sia = None
            self.stop_words = set()
        
        # Advanced keyword patterns for mental health conditions
        self.condition_patterns = {
            "depression": {
                "primary": ["depressed", "depression", "sad", "hopeless", "worthless", "empty"],
                "secondary": ["tired", "fatigue", "sleep", "appetite", "concentration", "guilt", "crying"],
                "behavioral": ["isolating", "withdrawn", "unmotivated", "lethargic", "lost interest"],
                "cognitive": ["negative thoughts", "self-criticism", "pessimistic", "can't think"]
            },
            "anxiety": {
                "primary": ["anxious", "anxiety", "worried", "panic", "fear", "nervous"],
                "secondary": ["restless", "tense", "on edge", "overwhelmed", "stress", "worry"],
                "physical": ["heart racing", "sweating", "trembling", "shortness of breath"],
                "behavioral": ["avoiding", "checking", "seeking reassurance", "can't relax"]
            },
            "bipolar": {
                "manic": ["manic", "high energy", "euphoric", "grandiose", "impulsive", "hyperactive"],
                "depressive": ["mood swings", "ups and downs", "extreme highs and lows", "cycling"],
                "mixed": ["rapid cycling", "mood episodes", "unstable mood", "emotional rollercoaster"]
            },
            "adhd": {
                "attention": ["can't focus", "distracted", "attention problems", "concentration issues"],
                "hyperactivity": ["hyperactive", "restless", "fidgety", "can't sit still", "always moving"],
                "impulsivity": ["impulsive", "act without thinking", "interrupt", "impatient"]
            },
            "ocd": {
                "obsessions": ["obsessive thoughts", "intrusive thoughts", "can't stop thinking", "stuck thoughts"],
                "compulsions": ["compulsive", "repetitive", "rituals", "checking", "counting", "organizing"],
                "interference": ["takes hours", "interferes with life", "can't function", "exhausting"]
            },
            "ptsd": {
                "trauma": ["trauma", "traumatic", "flashbacks", "nightmares", "reliving"],
                "avoidance": ["avoiding", "can't think about", "triggers", "reminders", "blocked out"],
                "hyperarousal": ["hypervigilant", "easily startled", "on guard", "jumpy", "can't sleep"]
            }
        }
        
        # Crisis detection patterns with severity levels
        self.crisis_patterns = {
            "immediate_danger": {
                "patterns": [
                    r"(?:want to|going to|plan to) (?:die|kill myself|end (?:it|my life))",
                    r"suicide plan|method to (?:die|kill)",
                    r"(?:tonight|today|now|soon) (?:I will|I'm going to) (?:die|kill myself)",
                    r"ready to (?:die|end it|kill myself)",
                ],
                "severity": 10
            },
            "suicidal_ideation": {
                "patterns": [
                    r"(?:think about|thoughts of) (?:dying|death|suicide|killing myself)",
                    r"wish I (?:was|were) dead",
                    r"life (?:isn't|is not) worth living",
                    r"(?:everyone|world) (?:would be|is) better (?:without|off without) me",
                    r"no (?:point|reason) (?:in |to )?(?:living|going on|continuing)"
                ],
                "severity": 8
            },
            "self_harm": {
                "patterns": [
                    r"(?:cut|cutting|harm|hurt) myself",
                    r"self.?harm|self.?injury",
                    r"want to hurt myself",
                    r"(?:burning|scratching|hitting) myself"
                ],
                "severity": 7
            },
            "severe_distress": {
                "patterns": [
                    r"can't (?:take|handle|go on|do) (?:this|it) anymore",
                    r"nothing (?:matters|helps|works)",
                    r"completely (?:hopeless|lost|broken)",
                    r"(?:give|giving) up on everything"
                ],
                "severity": 6
            },
            "help_seeking": {
                "patterns": [
                    r"need help",
                    r"don't know what to do",
                    r"(?:please|someone) help me",
                    r"can't do this alone"
                ],
                "severity": 4
            }
        }
        
        # Protective factors (reduce risk)
        self.protective_factors = [
            "support", "family", "friends", "therapy", "counselor", "help", "treatment",
            "hope", "future", "goals", "pets", "children", "faith", "religion", "medication"
        ]
        
    async def comprehensive_analysis(
        self, 
        assessment: MentalHealthAssessment,
        user_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive mental health analysis using multiple AI models"""
        
        try:
            text = assessment.text_input.lower()
            logger.info(f"🧠 Starting comprehensive analysis for {len(text)} character input")
            
            # Run multiple analyses concurrently
            tasks = [
                self._ai_sentiment_analysis(text),
                self._emotion_analysis(text),
                self._advanced_crisis_detection(text),
                self._condition_classification_ai(text),
                self._linguistic_analysis(text),
                self._contextual_analysis(text, user_history),
                self._protective_factors_analysis(text)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Safely extract results
            sentiment_result = results[0] if not isinstance(results[0], Exception) else {}
            emotion_result = results[1] if not isinstance(results[1], Exception) else {}
            crisis_result = results[2] if not isinstance(results[2], Exception) else None
            conditions_result = results[3] if not isinstance(results[3], Exception) else {}
            linguistic_result = results[4] if not isinstance(results[4], Exception) else {}
            contextual_result = results[5] if not isinstance(results[5], Exception) else {}
            protective_result = results[6] if not isinstance(results[6], Exception) else {}
            
            # Calculate final risk level
            risk_level = self._calculate_comprehensive_risk(
                crisis_result, sentiment_result, emotion_result, 
                conditions_result, linguistic_result, contextual_result, protective_result
            )
            
            # Generate personalized recommendations
            recommendations = self._generate_advanced_recommendations(
                risk_level, conditions_result, sentiment_result, emotion_result
            )
            
            # Extract crisis indicators
            crisis_indicators = crisis_result.get("crisis_indicators", []) if crisis_result else []
            
            result = {
                "risk_level": risk_level,
                "predicted_conditions": list(conditions_result.keys()) if conditions_result else [],
                "confidence_scores": conditions_result if conditions_result else {},
                "recommendations": recommendations,
                "crisis_indicators": crisis_indicators,
                "requires_immediate_attention": crisis_result.get("is_crisis", False) if crisis_result else False,
                "sentiment_analysis": sentiment_result,
                "emotion_analysis": emotion_result,
                "linguistic_patterns": linguistic_result,
                "protective_factors": protective_result,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "analysis_method": "comprehensive_multi_model"
            }
            
            logger.info(f"✅ Analysis completed: {risk_level} risk level, {len(crisis_indicators)} crisis indicators")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive analysis: {str(e)}")
            # Return safe default response
            return {
                "risk_level": "medium",  # Default to medium for safety
                "predicted_conditions": [],
                "confidence_scores": {},
                "recommendations": [
                    "Please consult with a mental health professional for personalized guidance.",
                    "If you're experiencing a mental health crisis, contact 988 (Crisis Lifeline).",
                    "Remember that seeking help is a sign of strength."
                ],
                "crisis_indicators": ["Analysis error occurred - recommend professional consultation"],
                "requires_immediate_attention": False,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "analysis_method": "fallback_safe_response"
            }
    
    async def _ai_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis using multiple approaches"""
        results = {}
        
        try:
            # VADER sentiment (rule-based, good for social media text)
            if self.sia:
                vader_scores = self.sia.polarity_scores(text)
                results["vader"] = {
                    "positive": vader_scores["pos"],
                    "negative": vader_scores["neg"],
                    "neutral": vader_scores["neu"],
                    "compound": vader_scores["compound"]
                }
            
            # TextBlob sentiment (statistical approach)
            blob = TextBlob(text)
            results["textblob"] = {
                "polarity": blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
            
            # Custom mental health sentiment
            results["mental_health_sentiment"] = self._mental_health_sentiment(text)
            
        except Exception as e:
            logger.warning(f"⚠️ Sentiment analysis error: {str(e)}")
            results = {"error": "sentiment_analysis_failed"}
        
        return results
    
    def _mental_health_sentiment(self, text: str) -> Dict[str, float]:
        """Custom sentiment analysis focused on mental health context"""
        
        # Mental health specific positive indicators
        positive_mh = ["better", "improving", "hopeful", "grateful", "progress", "healing", "support", "strength"]
        # Mental health specific negative indicators  
        negative_mh = ["worse", "deteriorating", "hopeless", "trapped", "isolated", "broken", "failing", "worthless"]
        # Neutral mental health terms
        neutral_mh = ["therapy", "medication", "treatment", "counseling", "assessment", "diagnosis"]
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_mh)
        neg_count = sum(1 for word in words if word in negative_mh)
        neu_count = sum(1 for word in words if word in neutral_mh)
        
        total = len(words)
        if total == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        return {
            "positive": pos_count / total,
            "negative": neg_count / total,
            "neutral": max(0.0, 1.0 - (pos_count + neg_count) / total)
        }
    
    async def _emotion_analysis(self, text: str) -> Dict[str, float]:
        """Emotion detection using keyword-based analysis"""
        
        emotion_keywords = {
            "joy": ["happy", "joyful", "excited", "pleased", "cheerful", "elated", "content"],
            "sadness": ["sad", "unhappy", "depressed", "melancholy", "grief", "sorrow", "heartbroken"],
            "anger": ["angry", "mad", "furious", "irritated", "annoyed", "rage", "frustrated"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried", "fearful", "panicked"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "startled"],
            "shame": ["ashamed", "embarrassed", "humiliated", "guilty", "mortified"],
            "contempt": ["contempt", "disdain", "scorn", "disgust", "hatred"]
        }
        
        words = text.lower().split()
        emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for word in words if any(keyword in word for keyword in keywords))
            if matches > 0:
                emotions[emotion] = min(matches / len(words) * 5, 1.0)  # Amplify and cap at 1.0
        
        return emotions
    
    async def _advanced_crisis_detection(self, text: str) -> Dict[str, Any]:
        """Advanced crisis detection using pattern matching and severity assessment"""
        
        crisis_indicators = []
        max_severity = 0
        detected_patterns = []
        
        try:
            # Pattern-based detection with severity scoring
            for category, config in self.crisis_patterns.items():
                category_detected = False
                for pattern in config["patterns"]:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        crisis_indicators.append(f"Crisis pattern detected: {category}")
                        detected_patterns.append({"category": category, "pattern": pattern, "matches": matches})
                        max_severity = max(max_severity, config["severity"])
                        category_detected = True
                        break  # Only count each category once
                
                if category_detected:
                    logger.warning(f"⚠️ Crisis pattern '{category}' detected with severity {config['severity']}")
            
            # Simple keyword detection for additional context
            crisis_keywords = [
                "suicide", "kill myself", "end it all", "want to die", "not worth living",
                "hopeless", "can't go on", "nothing matters", "everyone better without me",
                "give up", "no point", "can't take it", "want out"
            ]
            
            found_keywords = [keyword for keyword in crisis_keywords if keyword in text]
            if found_keywords:
                crisis_indicators.extend([f"Crisis keyword: {keyword}" for keyword in found_keywords])
                max_severity = max(max_severity, 5 + len(found_keywords))
            
            # Determine crisis level and actions
            is_crisis = max_severity >= 5
            professional_referral = max_severity >= 3
            emergency_contact = max_severity >= 8
            
            immediate_actions = []
            if emergency_contact:
                immediate_actions.extend([
                    "🚨 Contact emergency services (911) immediately if in physical danger",
                    "📞 Call or text 988 (Suicide & Crisis Lifeline) right now",
                    "👥 Do not leave the person alone - contact someone immediately",
                    "🏥 Consider going to the nearest emergency room"
                ])
            elif is_crisis:
                immediate_actions.extend([
                    "📞 Contact 988 (Suicide & Crisis Lifeline) for immediate support",
                    "📱 Text HOME to 741741 (Crisis Text Line)",
                    "👥 Reach out to a trusted friend, family member, or counselor",
                    "🏥 Consider contacting a mental health professional today"
                ])
            elif professional_referral:
                immediate_actions.extend([
                    "👨‍⚕️ Consider scheduling an appointment with a mental health professional",
                    "📞 Contact your doctor or a counseling service",
                    "👥 Reach out to your support network"
                ])
            
            result = {
                "is_crisis": is_crisis,
                "severity_level": min(max_severity, 10),
                "crisis_indicators": crisis_indicators,
                "detected_patterns": detected_patterns,
                "immediate_actions": immediate_actions,
                "professional_referral_needed": professional_referral,
                "emergency_contact_needed": emergency_contact,
                "analysis_confidence": min(len(crisis_indicators) / 3.0, 1.0)  # Confidence based on indicators
            }
            
            if is_crisis:
                logger.critical(f"🚨 CRISIS DETECTED - Severity: {max_severity}/10")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Crisis detection error: {str(e)}")
            # Return safe default - assume potential crisis for safety
            return {
                "is_crisis": True,
                "severity_level": 5,
                "crisis_indicators": ["Crisis detection system error - recommend professional consultation"],
                "immediate_actions": [
                    "📞 Contact 988 (Crisis Lifeline) for safety",
                    "👨‍⚕️ Consult with a mental health professional"
                ],
                "professional_referral_needed": True,
                "emergency_contact_needed": False
            }
    
    async def _condition_classification_ai(self, text: str) -> Dict[str, float]:
        """Advanced mental health condition classification"""
        
        condition_scores = {}
        
        try:
            for condition, pattern_groups in self.condition_patterns.items():
                total_score = 0
                total_possible = 0
                
                for category, keywords in pattern_groups.items():
                    category_matches = sum(1 for keyword in keywords if keyword in text)
                    
                    if category_matches > 0:
                        # Weight different categories differently
                        category_weights = {
                            "primary": 3.0,
                            "secondary": 2.0,
                            "physical": 2.0,
                            "behavioral": 1.5,
                            "cognitive": 1.5,
                            "manic": 2.5,
                            "depressive": 2.5,
                            "mixed": 2.0,
                            "attention": 2.0,
                            "hyperactivity": 2.0,
                            "impulsivity": 2.0,
                            "obsessions": 3.0,
                            "compulsions": 3.0,
                            "interference": 2.0,
                            "trauma": 3.0,
                            "avoidance": 2.0,
                            "hyperarousal": 2.0
                        }
                        
                        weight = category_weights.get(category, 1.0)
                        total_score += category_matches * weight
                    
                    total_possible += len(keywords) * category_weights.get(category, 1.0)
                
                if total_score > 0:
                    # Normalize score and apply threshold
                    normalized_score = min(total_score / (total_possible * 0.2), 1.0)
                    if normalized_score >= 0.15:  # Minimum threshold
                        condition_scores[condition] = normalized_score
            
            # Sort by confidence score
            condition_scores = dict(sorted(condition_scores.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"❌ Condition classification error: {str(e)}")
        
        return condition_scores
    
    async def _linguistic_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic patterns that may indicate mental health status"""
        
        analysis = {}
        
        try:
            # Basic text statistics
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if self.stop_words:
                words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
                content_words = [w for w in words if w not in self.stop_words]
            else:
                words = text.lower().split()
                content_words = words
            
            analysis["word_count"] = len(words)
            analysis["sentence_count"] = len(sentences)
            analysis["avg_words_per_sentence"] = len(words) / max(len(sentences), 1)
            analysis["content_word_ratio"] = len(content_words) / max(len(words), 1)
            
            # Pronoun usage analysis (self-reference patterns)
            first_person_pronouns = ["i", "me", "my", "myself", "mine"]
            first_person_count = sum(1 for word in words if word in first_person_pronouns)
            analysis["first_person_ratio"] = first_person_count / max(len(words), 1)
            
            # Negative language patterns
            negative_words = [
                "not", "no", "never", "nothing", "nobody", "nowhere", "neither", "nor",
                "can't", "won't", "shouldn't", "couldn't", "wouldn't", "isn't", "aren't", 
                "wasn't", "weren't", "don't", "doesn't", "didn't", "hasn't", "haven't", "hadn't"
            ]
            negative_count = sum(1 for word in words if word in negative_words)
            analysis["negative_word_ratio"] = negative_count / max(len(words), 1)
            
            # Absolutist thinking indicators
            absolutist_words = ["always", "never", "everyone", "no one", "everything", "nothing", "all", "none", "completely", "totally", "absolutely"]
            absolutist_count = sum(1 for word in words if word in absolutist_words)
            analysis["absolutist_ratio"] = absolutist_count / max(len(words), 1)
            
            # Time orientation analysis
            past_words = ["was", "were", "had", "did", "yesterday", "before", "ago", "earlier", "used to", "back then"]
            present_words = ["am", "is", "are", "now", "today", "currently", "right now", "at the moment"]
            future_words = ["will", "shall", "going", "tomorrow", "later", "soon", "next", "planning", "hope"]
            
            past_count = sum(1 for word in words if word in past_words)
            present_count = sum(1 for word in words if word in present_words)
            future_count = sum(1 for word in words if word in future_words)
            
            total_time_words = past_count + present_count + future_count
            if total_time_words > 0:
                analysis["time_orientation"] = {
                    "past": past_count / total_time_words,
                    "present": present_count / total_time_words,
                    "future": future_count / total_time_words
                }
            else:
                analysis["time_orientation"] = {"past": 0, "present": 0, "future": 0}
            
            # Emotional intensity indicators
            intensity_words = ["extremely", "very", "really", "so", "too", "incredibly", "terribly", "awful", "terrible"]
            intensity_count = sum(1 for word in words if word in intensity_words)
            analysis["emotional_intensity_ratio"] = intensity_count / max(len(words), 1)
            
        except Exception as e:
            logger.error(f"❌ Linguistic analysis error: {str(e)}")
            analysis = {"error": "linguistic_analysis_failed"}
        
        return analysis
    
    async def _contextual_analysis(self, text: str, user_history: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze text in context of user's conversation history"""
        
        context = {}
        
        try:
            if user_history:
                # Calculate conversation trends
                context["conversation_length"] = len(user_history)
                context["current_message_length"] = len(text.split())
                
                # Analyze message length trends
                if len(user_history) >= 3:
                    recent_lengths = [len(msg.split()) for msg in user_history[-3:]]
                    avg_length = sum(recent_lengths) / len(recent_lengths)
                    context["message_length_trend"] = "increasing" if len(text.split()) > avg_length * 1.2 else "decreasing" if len(text.split()) < avg_length * 0.8 else "stable"
                
                # Look for repeated themes or concerns
                all_text = " ".join(user_history + [text]).lower()
                common_words = [word for word, count in Counter(all_text.split()).most_common(10) if len(word) > 3]
                context["recurring_themes"] = common_words[:5]
                
                # Estimate engagement level
                context["engagement_level"] = min(len(text.split()) / 20, 1.0)  # Normalize to 0-1
                
        except Exception as e:
            logger.warning(f"⚠️ Contextual analysis error: {str(e)}")
        
        return context
    
    async def _protective_factors_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze protective factors that may reduce mental health risks"""
        
        protective_analysis = {}
        
        try:
            found_factors = []
            for factor in self.protective_factors:
                if factor in text.lower():
                    found_factors.append(factor)
            
            protective_analysis["protective_factors"] = found_factors
            protective_analysis["protective_score"] = len(found_factors) / len(self.protective_factors)
            protective_analysis["has_support_system"] = any(factor in ["support", "family", "friends"] for factor in found_factors)
            protective_analysis["has_professional_help"] = any(factor in ["therapy", "counselor", "treatment", "medication"] for factor in found_factors)
            protective_analysis["has_hope_factors"] = any(factor in ["hope", "future", "goals"] for factor in found_factors)
            
        except Exception as e:
            logger.warning(f"⚠️ Protective factors analysis error: {str(e)}")
            protective_analysis = {"protective_factors": [], "protective_score": 0.0}
        
        return protective_analysis
    
    def _calculate_comprehensive_risk(
        self,
        crisis_result: Optional[Dict],
        sentiment_result: Dict,
        emotion_result: Dict,
        conditions_result: Dict,
        linguistic_result: Dict,
        contextual_result: Dict,
        protective_result: Dict
    ) -> str:
        """Calculate comprehensive risk level from all analyses"""
        
        risk_score = 0
        
        try:
            # Crisis assessment (highest priority)
            if crisis_result and crisis_result.get("is_crisis"):
                severity = crisis_result.get("severity_level", 5)
                if severity >= 9:
                    return "critical"
                elif severity >= 7:
                    return "high"
                elif severity >= 5:
                    risk_score += 8
                else:
                    risk_score += 6
            
            # Sentiment analysis impact
            if sentiment_result:
                # VADER compound score
                vader_compound = sentiment_result.get("vader", {}).get("compound", 0)
                if vader_compound < -0.8:
                    risk_score += 5
                elif vader_compound < -0.5:
                    risk_score += 3
                elif vader_compound < -0.2:
                    risk_score += 1
                
                # TextBlob polarity
                textblob_polarity = sentiment_result.get("textblob", {}).get("polarity", 0)
                if textblob_polarity < -0.6:
                    risk_score += 4
                elif textblob_polarity < -0.3:
                    risk_score += 2
                
                # Mental health specific sentiment
                mh_negative = sentiment_result.get("mental_health_sentiment", {}).get("negative", 0)
                if mh_negative > 0.1:
                    risk_score += int(mh_negative * 10)
            
            # Emotion analysis impact
            if emotion_result:
                high_risk_emotions = ["sadness", "fear", "anger", "shame"]
                for emotion in high_risk_emotions:
                    if emotion in emotion_result:
                        score = emotion_result[emotion]
                        if score > 0.7:
                            risk_score += 3
                        elif score > 0.4:
                            risk_score += 2
                        elif score > 0.2:
                            risk_score += 1
            
            # Mental health conditions impact
            if conditions_result:
                severe_conditions = ["depression", "bipolar", "ptsd"]
                for condition, confidence in conditions_result.items():
                    if condition in severe_conditions:
                        if confidence > 0.7:
                            risk_score += 4
                        elif confidence > 0.5:
                            risk_score += 3
                        elif confidence > 0.3:
                            risk_score += 2
                    else:  # Other conditions
                        if confidence > 0.6:
                            risk_score += 2
                        elif confidence > 0.4:
                            risk_score += 1
            
            # Linguistic patterns impact
            if linguistic_result and not linguistic:
                if linguistic_result and "error" not in linguistic_result:
                    negative_ratio = linguistic_result.get("negative_word_ratio", 0)
                if negative_ratio > 0.2:
                    risk_score += 3
                elif negative_ratio > 0.15:
                    risk_score += 2
                elif negative_ratio > 0.1:
                    risk_score += 1
                
                absolutist_ratio = linguistic_result.get("absolutist_ratio", 0)
                if absolutist_ratio > 0.08:
                    risk_score += 2
                elif absolutist_ratio > 0.05:
                    risk_score += 1
                
                first_person_ratio = linguistic_result.get("first_person_ratio", 0)
                if first_person_ratio > 0.25:
                    risk_score += 2
                elif first_person_ratio > 0.2:
                    risk_score += 1
                
                time_orientation = linguistic_result.get("time_orientation", {})
                past_focus = time_orientation.get("past", 0)
                future_focus = time_orientation.get("future", 0)
                
                if past_focus > 0.6 and future_focus < 0.1:
                    risk_score += 2
                elif past_focus > 0.4:
                    risk_score += 1
                
                intensity_ratio = linguistic_result.get("emotional_intensity_ratio", 0)
                if intensity_ratio > 0.1:
                    risk_score += 1
            
            # Protective factors (reduce risk)
            if protective_result:
                protective_score = protective_result.get("protective_score", 0)
                if protective_score > 0.3:
                    risk_score -= 3
                elif protective_score > 0.2:
                    risk_score -= 2
                elif protective_score > 0.1:
                    risk_score -= 1
                
                if protective_result.get("has_support_system"):
                    risk_score -= 2
                if protective_result.get("has_professional_help"):
                    risk_score -= 2
                if protective_result.get("has_hope_factors"):
                    risk_score -= 1
            
            # Contextual factors
            if contextual_result:
                engagement_level = contextual_result.get("engagement_level", 0)
                if engagement_level < 0.3:
                    risk_score += 1
            
            # Determine final risk level
            risk_score = max(0, risk_score)
            
            if risk_score >= 15:
                return "critical"
            elif risk_score >= 10:
                return "high"
            elif risk_score >= 5:
                return "medium"
            else:
                return "low"
        
        except Exception as e:
            logger.error(f"❌ Risk calculation error: {str(e)}")
            return "medium"
    
    def _generate_advanced_recommendations(
        self,
        risk_level: str,
        conditions: Dict[str, float],
        sentiment: Dict[str, Any],
        emotions: Dict[str, float]
    ) -> List[str]:
        """Generate personalized recommendations"""
        
        recommendations = []
        
        try:
            # Base recommendations by risk level
            base_recommendations = {
                "critical": [
                    "🚨 IMMEDIATE ACTION: Contact 911 or Crisis Lifeline 988",
                    "📞 Call 988 Suicide & Crisis Lifeline immediately",
                    "💬 Text HOME to 741741 for crisis support",
                    "👥 Do not be alone - contact someone now",
                    "🏥 Go to nearest emergency room"
                ],
                "high": [
                    "📞 Contact 988 Crisis Lifeline today",
                    "👨‍⚕️ Schedule emergency mental health appointment",
                    "👥 Reach out to friends/family for support",
                    "📱 Download crisis support apps"
                ],
                "medium": [
                    "👨‍⚕️ Schedule mental health appointment this week",
                    "🧘 Practice daily self-care and stress management",
                    "👥 Connect with support network regularly",
                    "📚 Learn coping strategies"
                ],
                "low": [
                    "🧘 Continue good mental health habits",
                    "👥 Maintain social connections",
                    "🎯 Set realistic goals",
                    "💭 Practice mindfulness daily"
                ]
            }
            
            recommendations.extend(base_recommendations.get(risk_level, base_recommendations["medium"]))
            
            # Condition-specific recommendations
            for condition, confidence in conditions.items():
                if confidence > 0.5:
                    if condition == "depression":
                        recommendations.extend([
                            "☀️ Get daily sunlight exposure",
                            "🏃‍♂️ Regular physical activity",
                            "📅 Maintain daily routine"
                        ])
                    elif condition == "anxiety":
                        recommendations.extend([
                            "🫁 Practice 4-7-8 breathing technique",
                            "🧘‍♀️ Try progressive muscle relaxation",
                            "☕ Limit caffeine intake"
                        ])
                    elif condition == "adhd":
                        recommendations.extend([
                            "📋 Use organizational tools",
                            "⏰ Break tasks into smaller steps",
                            "🏃‍♂️ Regular exercise improves focus"
                        ])
            
            # General recommendations
            recommendations.extend([
                "💤 Prioritize 7-9 hours of sleep nightly",
                "🥗 Maintain balanced diet and hydration",
                "📚 Learn about mental health resources",
                "💪 Remember: seeking help shows strength"
            ])
            
            return recommendations[:12]
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {str(e)}")
            return [
                "Please consult a mental health professional",
                "Crisis support: 988 or text HOME to 741741"
            ]

# Create analyzer instance
advanced_analyzer = AdvancedMentalHealthAnalyzer()

# Utility function
async def analyze_mental_health_text(text_input: str, user_history: Optional[List[str]] = None) -> Dict[str, Any]:
    """Utility function to analyze mental health text"""
    try:
        assessment = type('MockAssessment', (), {
            'text_input': text_input,
            'assessment_type': 'general',
            'previous_context': user_history
        })()
        
        result = await advanced_analyzer.comprehensive_analysis(assessment, user_history)
        logger.info(f"✅ Analysis completed for {len(text_input)} character input")
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return {
            "error": "analysis_failed",
            "message": str(e),
            "fallback_recommendations": [
                "Please consult with a mental health professional",
                "Crisis support: 988 or text HOME to 741741"
            ]
        }

# Export
__all__ = ['advanced_analyzer', 'AdvancedMentalHealthAnalyzer', 'analyze_mental_health_text']