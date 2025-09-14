# =============================================================================
# Enhanced Chat Service - Better Conversational AI
# This replaces the basic chat service with more intelligent responses
# =============================================================================

import random
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from dataclasses import dataclass, field
from loguru import logger

# Import schemas (adjust as needed for your project)
try:
    from app.models.schemas import ChatMessage, ChatResponse, MessageType, RiskLevel
except ImportError:
    # Fallback if schemas not available
    logger.warning("Using fallback chat schemas")
    
    class MessageType:
        USER = "user"
        AI = "ai"
        SYSTEM = "system"
    
    class RiskLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

@dataclass
class ConversationContext:
    """Enhanced context tracking for conversations"""
    user_id: str
    current_mood: Optional[str] = None
    conversation_stage: str = "greeting"
    topics_discussed: List[str] = field(default_factory=list)
    risk_level: str = "low"
    last_assessment_result: Optional[Dict] = None
    session_start: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    user_name: Optional[str] = None
    conversation_flow: List[str] = field(default_factory=list)

class EnhancedConversationalAI:
    """Advanced conversational AI for mental health support with much better responses"""
    
    def __init__(self):
        self.conversation_contexts = {}
        self.initialize_response_system()
        logger.info("✅ Enhanced Conversational AI initialized")
    
    def initialize_response_system(self):
        """Initialize comprehensive response templates"""
        
        # Greeting responses with variety
        self.greeting_responses = [
            "Hello! I'm really glad you're here. It takes courage to reach out, and I want you to know this is a completely safe space. What's been on your mind lately?",
            "Hi there! Thank you for connecting with me today. I'm here to listen without judgment and provide support however I can. How are you feeling right now?",
            "Welcome! I'm so pleased you've decided to talk. Everyone deserves to have their feelings heard and validated. What would you like to share today?",
            "Hello, and thank you for trusting me with your time today. I'm here to offer support and understanding. What's been weighing on your heart recently?",
            "Hi! I'm honored that you've chosen to open up here. This is your space to express whatever you're experiencing. How can I best support you today?",
            "Good to meet you! I want you to know that whatever you're going through, you don't have to face it alone. What's been challenging for you lately?",
            "Hello! I'm here as a caring listener and supporter. Every feeling you have is valid, and I'm here to help you work through whatever is on your mind. What's been troubling you?"
        ]
        
        # Empathetic responses for different emotions
        self.empathetic_responses = {
            "sadness": [
                "I can really hear the sadness in what you're sharing, and I want you to know that it's completely okay to feel this way. Sadness is a natural human emotion, even though it's painful.",
                "It sounds like you're carrying a heavy emotional burden right now. That kind of deep sadness can feel overwhelming, and I'm here to help you process these feelings.",
                "Thank you for being so open about your sadness. It takes strength to acknowledge and express these difficult emotions. You're not alone in feeling this way.",
                "I can sense how much pain you're in right now. Sadness can feel all-consuming, but please know that these feelings, while valid and important, don't define your entire future.",
                "Your sadness is being heard and acknowledged. It's okay to sit with these feelings - they're telling you something important about what matters to you."
            ],
            "anxiety": [
                "I understand that anxiety can make everything feel more intense and overwhelming. What you're experiencing with these worried thoughts and feelings is very real and valid.",
                "Anxiety has a way of making our minds race with 'what if' scenarios. It sounds like you're dealing with a lot of mental pressure right now, and that's exhausting.",
                "Thank you for sharing about your anxiety. It takes courage to talk about those racing thoughts and worried feelings. You're taking a positive step by reaching out.",
                "I can hear how anxiety is affecting you. Those feelings of being on edge or overwhelmed are your mind's way of trying to protect you, even though they don't feel helpful right now.",
                "Anxiety can make even small things feel monumental. I want you to know that what you're experiencing is common, treatable, and you don't have to navigate this alone."
            ],
            "anger": [
                "I can sense your frustration and anger, and those feelings are completely valid. Anger often comes up when we feel hurt, misunderstood, or when our boundaries have been crossed.",
                "Thank you for being honest about your anger. It's actually a healthy sign that you can recognize and express these intense feelings rather than bottling them up.",
                "Anger can be such a powerful emotion, and it sounds like you're dealing with some really difficult situations. Your feelings make complete sense given what you're going through.",
                "I hear your frustration, and it's important to acknowledge that anger often masks other emotions like hurt, disappointment, or feeling powerless. What do you think might be underneath these angry feelings?",
                "Your anger is telling you something important - maybe that something isn't fair, that your needs aren't being met, or that you need to set some boundaries. Let's explore what it's trying to communicate."
            ],
            "fear": [
                "Fear can be such an overwhelming emotion, making everything feel uncertain and unsafe. I want you to know that your fears are valid, even if they might seem irrational to others.",
                "I can hear how frightened you're feeling. Fear has a way of making our world feel smaller and more dangerous. You're being very brave by talking about these scary feelings.",
                "Thank you for sharing about your fears with me. It takes real courage to be vulnerable about what scares us most. You don't have to face these fears alone.",
                "Fear can hijack our thinking and make everything seem more threatening. What you're experiencing is your mind's way of trying to keep you safe, even though it doesn't feel helpful.",
                "I understand that fear can make decision-making feel impossible. Let's work together to understand what's driving these fears and how we can help you feel more secure."
            ],
            "overwhelmed": [
                "It sounds like you have so much on your plate right now that it's hard to see a way forward. Feeling overwhelmed is your mind's signal that you're dealing with more than feels manageable.",
                "I can hear how much you're juggling right now. When everything feels urgent and important, it's natural to feel paralyzed about where to even begin.",
                "Thank you for sharing how overwhelmed you're feeling. Sometimes life throws more at us than we feel equipped to handle, and that's not a reflection of your strength or capability.",
                "Overwhelm can make everything feel impossible and urgent at the same time. Let's see if we can break things down into smaller, more manageable pieces together.",
                "I understand that feeling of being pulled in too many directions. It's like having too many browser tabs open in your mind - everything is running but nothing is getting your full attention."
            ],
            "loneliness": [
                "Loneliness can be one of the most painful emotions because humans are naturally social beings. I want you to know that feeling lonely doesn't mean you're alone in the world.",
                "I hear how isolated you're feeling right now. Loneliness isn't just about being physically alone - it's about feeling disconnected or misunderstood, even when people are around.",
                "Thank you for trusting me with these feelings of loneliness. It takes courage to admit when we're feeling disconnected from others. You've taken a step toward connection by reaching out here.",
                "Loneliness can make us feel like we're the only ones struggling, but the truth is that most people experience deep loneliness at some point. You're more connected to the human experience than you might realize.",
                "I can sense how much you're longing for genuine connection and understanding. That desire for meaningful relationships is actually a beautiful part of being human."
            ]
        }
        
        # Follow-up questions to deepen conversation
        self.follow_up_questions = [
            "Can you tell me more about what that experience was like for you?",
            "How long have you been feeling this way?",
            "What do you think might have triggered these feelings?",
            "How are these feelings affecting your daily life?",
            "What does a typical day look like for you when you're feeling this way?",
            "Have you noticed any patterns in when these feelings are strongest?",
            "What would you want someone who cares about you to know about how you're feeling?",
            "If you could change one thing about your current situation, what would it be?",
            "What has helped you get through difficult times in the past?",
            "How would you describe your support system right now?",
            "What brings you even small moments of peace or comfort?",
            "What would feeling better look like to you?"
        ]
        
        # Supportive affirmations
        self.affirmations = [
            "Your feelings are completely valid and understandable.",
            "You're showing real strength by being so open and honest.",
            "Thank you for trusting me with something so personal.",
            "What you're going through sounds really challenging.",
            "You don't have to have all the answers right now.",
            "Taking things one day at a time is perfectly okay.",
            "You're not broken, even though you might feel that way sometimes.",
            "Your emotions make sense given what you're experiencing.",
            "You deserve compassion and understanding, especially from yourself.",
            "Healing isn't linear, and that's completely normal."
        ]
        
        # Therapeutic techniques and coping strategies
        self.coping_strategies = {
            "anxiety": [
                "Try the 4-7-8 breathing technique: breathe in for 4 counts, hold for 7, breathe out for 8. This can help activate your body's relaxation response.",
                "The 5-4-3-2-1 grounding technique can help when anxiety feels overwhelming: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                "Progressive muscle relaxation can help release physical tension that comes with anxiety. Start with your toes and work your way up, tensing and then relaxing each muscle group.",
                "Sometimes anxiety thrives on 'what if' thinking. Try asking yourself: 'What is actually happening right now?' to bring your focus back to the present moment."
            ],
            "depression": [
                "Even small activities can help when depression feels heavy. Could you try one tiny thing today, like making your bed or drinking a glass of water?",
                "Depression often tells us lies about our worth and future. What would you tell a good friend who was feeling exactly like you are right now?",
                "Sometimes when we're depressed, we stop doing things we used to enjoy. Is there one small activity that used to bring you pleasure that you might try for just a few minutes?",
                "Sunlight can sometimes help with mood. If possible, try to spend even a few minutes outside or near a window with natural light."
            ],
            "stress": [
                "When stress builds up, our bodies hold tension. Try doing some gentle neck rolls or shoulder shrugs to release physical stress.",
                "Stress often comes from feeling like we have to do everything at once. What's one thing you could take off your plate or ask for help with?",
                "Sometimes writing down everything we're worried about can help get it out of our heads and onto paper where it feels more manageable.",
                "Taking short breaks throughout the day, even just 60 seconds to breathe deeply, can help prevent stress from building up."
            ]
        }
        
        # Crisis keywords for detection
        self.crisis_keywords = [
            "suicide", "kill myself", "end it all", "want to die", "not worth living",
            "hopeless", "can't go on", "nothing matters", "everyone better without me",
            "give up", "no point", "can't take it", "want out", "end my life"
        ]
    
    async def generate_response(
        self, 
        user_message: str, 
        user_id: str, 
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate contextually appropriate response"""
        
        try:
            # Get or create conversation context
            context = self.conversation_contexts.get(user_id)
            if not context:
                context = ConversationContext(user_id=user_id)
                self.conversation_contexts[user_id] = context
            
            context.message_count += 1
            
            # Analyze the message
            analysis = await self._analyze_user_message(user_message, context)
            
            # Update context
            context.current_mood = analysis.get("dominant_emotion")
            context.conversation_stage = analysis.get("conversation_stage")
            context.risk_level = analysis.get("risk_level", "low")
            
            # Generate appropriate response
            if analysis.get("is_crisis"):
                response = await self._generate_crisis_response(user_message, context, analysis)
            else:
                response = await self._generate_contextual_response(user_message, context, analysis)
            
            # Add to conversation flow
            context.conversation_flow.append(f"user: {user_message[:50]}...")
            context.conversation_flow.append(f"ai: {response['ai_response'][:50]}...")
            
            # Keep conversation flow manageable
            if len(context.conversation_flow) > 20:
                context.conversation_flow = context.conversation_flow[-20:]
            
            logger.info(f"💬 Generated response for user {user_id}, message #{context.message_count}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            return {
                "ai_response": "I'm here to listen and support you. Could you tell me more about how you're feeling right now? If you're in crisis, please contact 988 for immediate help.",
                "suggested_resources": ["Crisis support: 988 or text HOME to 741741"],
                "requires_human_intervention": False
            }
    
    async def _analyze_user_message(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze user message for emotion, intent, and risk"""
        
        message_lower = message.lower()
        analysis = {}
        
        # Crisis detection
        is_crisis = any(keyword in message_lower for keyword in self.crisis_keywords)
        analysis["is_crisis"] = is_crisis
        analysis["risk_level"] = "critical" if is_crisis else "low"
        
        # Emotion detection
        emotions = self._detect_emotions(message_lower)
        analysis["dominant_emotion"] = emotions[0] if emotions else "neutral"
        analysis["detected_emotions"] = emotions
        
        # Conversation stage
        if context.message_count <= 2:
            if any(greeting in message_lower for greeting in ["hi", "hello", "hey"]):
                analysis["conversation_stage"] = "greeting"
            else:
                analysis["conversation_stage"] = "opening"
        elif context.message_count > 10:
            analysis["conversation_stage"] = "deep_conversation"
        else:
            analysis["conversation_stage"] = "building_rapport"
        
        # Intent detection
        analysis["user_intent"] = self._detect_intent(message_lower)
        
        return analysis
    
    def _detect_emotions(self, message: str) -> List[str]:
        """Detect emotions in user message"""
        
        emotion_patterns = {
            "sadness": ["sad", "depressed", "down", "crying", "tears", "heartbroken", "devastated", "miserable"],
            "anxiety": ["anxious", "worried", "nervous", "panic", "scared", "afraid", "overwhelmed", "stressed"],
            "anger": ["angry", "mad", "furious", "frustrated", "annoyed", "irritated", "rage", "hate"],
            "fear": ["scared", "afraid", "terrified", "fearful", "frightened", "worried", "anxious"],
            "loneliness": ["lonely", "alone", "isolated", "disconnected", "abandoned", "empty"],
            "overwhelmed": ["overwhelmed", "too much", "can't handle", "drowning", "buried", "swamped"],
            "hopeless": ["hopeless", "pointless", "no point", "give up", "useless", "worthless"]
        }
        
        detected_emotions = []
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in message for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions
    
    def _detect_intent(self, message: str) -> str:
        """Detect user's intent or what they're looking for"""
        
        if any(word in message for word in ["help", "advice", "what should i do", "don't know"]):
            return "seeking_advice"
        elif any(word in message for word in ["listen", "talk", "share", "tell you"]):
            return "wanting_to_share"
        elif any(word in message for word in ["better", "feel good", "improve", "change"]):
            return "wanting_improvement"
        elif any(word in message for word in ["why", "understand", "confused", "don't get it"]):
            return "seeking_understanding"
        else:
            return "general_support"
    
    async def _generate_contextual_response(
        self, 
        user_message: str, 
        context: ConversationContext, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate contextual response based on analysis"""
        
        stage = analysis.get("conversation_stage", "general")
        dominant_emotion = analysis.get("dominant_emotion", "neutral")
        user_intent = analysis.get("user_intent", "general_support")
        
        # Start building response
        response_parts = []
        
        # 1. Empathetic acknowledgment
        if dominant_emotion in self.empathetic_responses:
            empathy = random.choice(self.empathetic_responses[dominant_emotion])
            response_parts.append(empathy)
        elif stage == "greeting":
            empathy = random.choice(self.greeting_responses)
            response_parts.append(empathy)
        else:
            # General empathetic responses
            general_empathy = [
                "I really appreciate you sharing that with me. It takes courage to open up about personal struggles.",
                "Thank you for trusting me with these feelings. What you're experiencing sounds really challenging.",
                "I can hear that this is weighing heavily on you. Your feelings are completely valid and understandable.",
                "It sounds like you're dealing with a lot right now. I'm here to listen and support you through this.",
                "I want you to know that what you're sharing is being heard with complete understanding and without judgment."
            ]
            response_parts.append(random.choice(general_empathy))
        
        # 2. Add validation/affirmation
        if context.message_count > 2:  # After initial greeting
            affirmation = random.choice(self.affirmations)
            response_parts.append(affirmation)
        
        # 3. Follow-up question or coping strategy
        if user_intent == "seeking_advice" and dominant_emotion in self.coping_strategies:
            coping_strategy = random.choice(self.coping_strategies[dominant_emotion])
            response_parts.append(f"Here's something that might help: {coping_strategy}")
        elif user_intent != "wanting_to_share":  # Don't interrupt if they just want to share
            follow_up = random.choice(self.follow_up_questions)
            response_parts.append(follow_up)
        
        # Combine response parts
        ai_response = "\n\n".join(response_parts)
        
        # Generate contextual resources
        resources = self._generate_contextual_resources(dominant_emotion, user_intent)
        
        return {
            "ai_response": ai_response,
            "suggested_resources": resources,
            "requires_human_intervention": False,
            "analysis_summary": {
                "emotion": dominant_emotion,
                "stage": stage,
                "intent": user_intent
            }
        }
    
    async def _generate_crisis_response(
        self, 
        user_message: str, 
        context: ConversationContext, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate crisis intervention response"""
        
        crisis_responses = [
            "I'm very concerned about what you've shared with me, and I want you to know that your life has tremendous value. The pain you're feeling right now is real, but it can change with proper support.",
            "Thank you for trusting me with these incredibly difficult feelings. What you're experiencing sounds overwhelming, but please know that there are people specially trained to help you through this crisis.",
            "I can hear how much emotional pain you're in right now. These feelings are a sign that you need immediate professional support, not that your situation is hopeless.",
            "The thoughts you're having about ending your life tell me that you're suffering greatly. I want to connect you with people who can provide the immediate, specialized help you deserve right now."
        ]
        
        response = random.choice(crisis_responses)
        
        crisis_resources = [
            "🚨 Immediate Help: Call or text 988 (Suicide & Crisis Lifeline) - available 24/7 with trained counselors",
            "📱 Text Support: Text HOME to 741741 (Crisis Text Line) for immediate text-based crisis counseling",
            "🚑 Emergency: If you're in immediate physical danger, please call 911",
            "🌐 Online Chat: Visit suicidepreventionlifeline.org for web chat with crisis counselors",
            "🏥 Local Resources: Consider going to your nearest emergency room where mental health professionals can help"
        ]
        
        full_response = f"{response}\n\nImmediate Crisis Resources:\n" + "\n".join(crisis_resources)
        
        return {
            "ai_response": full_response,
            "suggested_resources": crisis_resources,
            "requires_human_intervention": True,
            "crisis_level": "high"
        }
    
    def _generate_contextual_resources(self, emotion: str, intent: str) -> List[str]:
        """Generate resources based on emotion and intent"""
        
        resources = []
        
        # Emotion-specific resources
        emotion_resources = {
            "anxiety": [
                "Practice box breathing: inhale 4, hold 4, exhale 4, hold 4",
                "Try the STOP technique: Stop, Take a breath, Observe, Proceed mindfully",
                "Anxiety & Depression Association of America: adaa.org"
            ],
            "depression": [
                "Even small steps matter - try one tiny positive action today",
                "Depression support groups: nami.org/support",
                "Consider reaching out to a mental health professional"
            ],
            "loneliness": [
                "Connection starts with self-compassion",
                "Look into local support groups or volunteer opportunities",
                "Online communities can provide meaningful connection"
            ],
            "overwhelmed": [
                "Break large problems into smaller, manageable steps",
                "It's okay to ask for help - you don't have to do everything alone",
                "Consider what you can postpone or delegate"
            ]
        }
        
        if emotion in emotion_resources:
            resources.extend(emotion_resources[emotion])
        
        # Intent-specific resources
        if intent == "seeking_advice":
            resources.extend([
                "Professional counselors are trained specifically to help with these challenges",
                "Every situation is unique - personalized guidance can make a real difference"
            ])
        elif intent == "wanting_improvement":
            resources.extend([
                "Change is possible with the right support and strategies",
                "Small, consistent steps often lead to meaningful improvements"
            ])
        
        # General supportive resources
        resources.extend([
            "Mental health is health - taking care of it is essential",
            "You deserve support, understanding, and professional help if needed"
        ])
        
        return resources[:4]  # Limit to most relevant
    
    def clear_conversation(self, user_id: str):
        """Clear conversation context for a user"""
        if user_id in self.conversation_contexts:
            del self.conversation_contexts[user_id]
            logger.info(f"🗑️ Cleared conversation context for user {user_id}")

# Create enhanced chat service instance
enhanced_chat_service = EnhancedConversationalAI()

# Export for use in routes
__all__ = ['enhanced_chat_service', 'EnhancedConversationalAI', 'ConversationContext']