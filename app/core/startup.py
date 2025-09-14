from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
import sys

from app.core.config import settings
from app.core.database import engine, Base
from app.services.integrated_mental_health import integrated_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan with AI service initialization"""
    
    # Startup
    logger.info("🚀 Starting Mental Health AI System v1.0.0")
    
    try:
        # Create database tables
        logger.info("📊 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database ready")
        
        # Initialize AI services
        logger.info("🤖 Initializing AI services...")
        await integrated_service.initialize()
        logger.info("✅ AI services ready")
        
        # Verify system status
        status = integrated_service.get_service_status()
        logger.info(f"📋 System Status: {status['status']}")
        
        # Log capabilities
        capabilities = status['capabilities']
        active_features = [feature for feature, active in capabilities.items() if active]
        logger.info(f"⚡ Active Features: {', '.join(active_features)}")
        
        logger.info("🎯 Mental Health AI System ready to help!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Mental Health AI System")
    logger.info("💙 Thank you for supporting mental health technology!")