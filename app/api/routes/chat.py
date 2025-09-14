from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger
from app.services.ai_models import ai_models


from app.core.database import get_db
from app.models.user import User
from app.models.schemas import (
    ChatMessage,
    ChatResponse,
    ConversationHistory,
    APIResponse,
    MessageType
)
from app.api.routes.auth import get_current_user

router = APIRouter()

class ChatService:
    """Chat service for mental health conversations"""
    
    def __init__(self, db: Session):
        self.db = db
        # In-memory storage for now (would be database in production)
        self.conversations = {}
    
    async def process_message(self, user: User, message: ChatMessage) -> ChatResponse:
        """Process user message and generate AI response"""
        try:
            # Store user message
            user_id = str(user.id)
            if user_id not in self.conversations:
                self.conversations[user_id] = []
            
            # Add user message to conversation
            message.timestamp = datetime.utcnow()
            self.conversations[user_id].append(message)
            
            # Generate AI response (simplified for now)
            ai_response = await self._generate_ai_response(message.content, user)
            
            # Store AI response
            ai_message = ChatMessage(
                content=ai_response.message,
                message_type=MessageType.AI,
                timestamp=datetime.utcnow()
            )
            self.conversations[user_id].append(ai_message)
            
            logger.info(f"Chat interaction processed for user {user.id}")
            return ai_response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process message"
            )
    
    async def _generate_ai_response(self, user_message: str, user: User) -> ChatResponse:
        """Generate AI response based on user message"""
        # This is a simplified response generator
        # In Day 4-5, we'll integrate with proper NLP models
        
        message_lower = user_message.lower()
        
        # Crisis keywords check
        crisis_keywords = ["suicide", "kill myself", "end it all", "want to die"]
        if any(keyword in message_lower for keyword in crisis_keywords):
            return ChatResponse(
                message="I'm very concerned about what you're sharing. Your life has value and there are people who want to help. Please contact the 988 Suicide & Crisis Lifeline immediately, or if you're in immediate danger, call 911. You don't have to go through this alone.",
                requires_human_intervention=True,
                suggested_resources=[
                    "988 Suicide & Crisis Lifeline: Call or text 988",
                    "Crisis Text Line: Text HOME to 741741",
                    "Emergency Services: Call 911"
                ]
            )
        
        # Greeting responses
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return ChatResponse(
                message=f"Hello {user.first_name or user.username}! I'm here to listen and provide support. How are you feeling today?",
                suggested_resources=[
                    "Take a moment to reflect on your emotions",
                    "Practice deep breathing if you're feeling stressed"
                ]
            )
        
        # Sad/depressed responses
        if any(word in message_lower for word in ["sad", "depressed", "down", "hopeless"]):
            return ChatResponse(
                message="I hear that you're going through a difficult time. It takes courage to share these feelings. Can you tell me more about what's been weighing on you lately?",
                suggested_resources=[
                    "Consider speaking with a counselor or therapist",
                    "Reach out to a trusted friend or family member",
                    "Try gentle activities like walking or listening to music"
                ]
            )
        
        # Anxious responses
        if any(word in message_lower for word in ["anxious", "worried", "panic", "stress"]):
            return ChatResponse(
                message="Anxiety can feel overwhelming, but you're taking a positive step by talking about it. Let's try to understand what's causing these feelings. What situations tend to trigger your anxiety?",
                suggested_resources=[
                    "Practice the 4-7-8 breathing technique",
                    "Try progressive muscle relaxation",
                    "Consider mindfulness or meditation apps"
                ]
            )
        
       
        
        # Default supportive response
        return ChatResponse(
            message="Thank you for sharing that with me. I'm here to listen and support you. Can you tell me more about how you're feeling right now?",
            suggested_resources=[
                "Remember that it's okay to have difficult emotions",
                "Consider journaling about your thoughts and feelings"
            ]
        )
    
    def get_conversation_history(self, user: User, limit: int = 50) -> ConversationHistory:
        """Get user's conversation history"""
        user_id = str(user.id)
        messages = self.conversations.get(user_id, [])
        
        # Limit messages if specified
        if limit and len(messages) > limit:
            messages = messages[-limit:]
        
        # Calculate summary (simplified)
        total_messages = len(messages)
        conversation_start = messages[0].timestamp if messages else datetime.utcnow()
        
        return ConversationHistory(
            messages=messages,
            summary=f"Conversation with {total_messages} messages",
            total_messages=total_messages,
            conversation_start=conversation_start
        )

@router.post("/message", response_model=APIResponse)
async def send_message(
    message: ChatMessage,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message to the AI chatbot"""
    try:
        chat_service = ChatService(db)
        response = await chat_service.process_message(current_user, message)
        
        return APIResponse(
            success=True,
            message="Message processed successfully",
            data={
                "ai_response": response.message,
                "suggested_resources": response.suggested_resources,
                "requires_human_intervention": response.requires_human_intervention,
                "timestamp": datetime.utcnow()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat message error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )

@router.get("/history", response_model=APIResponse)
async def get_chat_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's chat history"""
    try:
        chat_service = ChatService(db)
        history = chat_service.get_conversation_history(current_user, limit)
        
        # Convert to serializable format
        messages_data = []
        for msg in history.messages:
            messages_data.append({
                "content": msg.content,
                "message_type": msg.message_type.value,
                "timestamp": msg.timestamp
            })
        
        return APIResponse(
            success=True,
            message="Chat history retrieved",
            data={
                "messages": messages_data,
                "total_messages": history.total_messages,
                "conversation_start": history.conversation_start,
                "summary": history.summary
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history"
        )

@router.delete("/history", response_model=APIResponse)
async def clear_chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clear user's chat history"""
    try:
        chat_service = ChatService(db)
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

@router.get("/conversation-summary", response_model=APIResponse)
async def get_conversation_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-generated conversation summary and insights"""
    try:
        chat_service = ChatService(db)
        user_id = str(current_user.id)
        
        messages = chat_service.conversations.get(user_id, [])
        
        if not messages:
            return APIResponse(
                success=True,
                message="No conversation history found",
                data={
                    "summary": "No conversations yet",
                    "insights": [],
                    "mood_trends": [],
                    "recommendations": ["Start a conversation to get personalized insights"]
                }
            )
        
        # Generate summary and insights (simplified version)
        user_messages = [msg for msg in messages if msg.message_type == MessageType.USER]
        
        # Basic sentiment analysis
        positive_words = ["good", "better", "happy", "grateful", "hopeful", "calm"]
        negative_words = ["sad", "anxious", "worried", "depressed", "hopeless", "angry"]
        
        mood_analysis = {"positive": 0, "negative": 0, "neutral": 0}
        
        for msg in user_messages:
            content_lower = msg.content.lower()
            has_positive = any(word in content_lower for word in positive_words)
            has_negative = any(word in content_lower for word in negative_words)
            
            if has_positive and not has_negative:
                mood_analysis["positive"] += 1
            elif has_negative and not has_positive:
                mood_analysis["negative"] += 1
            else:
                mood_analysis["neutral"] += 1
        
        # Generate insights
        insights = []
        if mood_analysis["negative"] > mood_analysis["positive"]:
            insights.append("Recent conversations show you may be experiencing some challenges")
            insights.append("Consider reaching out for additional support")
        elif mood_analysis["positive"] > mood_analysis["negative"]:
            insights.append("You've shared some positive moments in our conversations")
            insights.append("Keep focusing on the things that bring you joy")
        
        total_messages = len(user_messages)
        insights.append(f"You've engaged in {total_messages} meaningful exchanges")
        
        return APIResponse(
            success=True,
            message="Conversation summary generated",
            data={
                "summary": f"Analyzed {len(messages)} total messages across your conversations",
                "insights": insights,
                "mood_trends": mood_analysis,
                "total_conversations": len(user_messages),
                "last_conversation": messages[-1].timestamp if messages else None,
                "recommendations": [
                    "Continue regular check-ins with yourself",
                    "Practice self-compassion in difficult moments",
                    "Remember that seeking support is a sign of strength"
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating conversation summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate conversation summary"
        )
