import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    AutoModel
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import asyncio
import pickle
import os
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
import json

from app.core.config import settings
from app.models.schemas import RiskLevel, MentalHealthCondition

class AIModelManager:
    """Central manager for all AI models used in mental health analysis"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.embeddings_model = None
        self.is_initialized = False
        
        # Model configurations
        self.model_configs = {
            "sentiment": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "task": "sentiment-analysis"
            },
            "mental_health": {
                "model_name": "mental/mental-bert-base-uncased",
                "task": "text-classification"
            },
            "emotion": {
                "model_name": "j-hartmann/emotion-english-distilroberta-base",
                "task": "text-classification"
            },
            "crisis": {
                "model_name": "martin-ha/toxic-comment-model",
                "task": "text-classification"
            }
        }
        
        # Embeddings for semantic similarity
        self.embeddings_model_name = "all-MiniLM-L6-v2"
        
    async def initialize_models(self):
        """Initialize all AI models asynchronously"""
        try:
            logger.info("🤖 Initializing AI models...")
            
            # Initialize sentiment analysis
            await self._initialize_sentiment_model()
            
            # Initialize mental health classification
            await self._initialize_mental_health_model()
            
            # Initialize emotion detection
            await self._initialize_emotion_model()
            
            # Initialize crisis detection
            await self._initialize_crisis_model()
            
            # Initialize embeddings model
            await self._initialize_embeddings_model()
            
            self.is_initialized = True
            logger.info("✅ All AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI models: {str(e)}")
            raise
    
    async def _initialize_sentiment_model(self):
        """Initialize sentiment analysis model"""
        try:
            config = self.model_configs["sentiment"]
            self.pipelines["sentiment"] = pipeline(
                config["task"],
                model=config["model_name"],
                return_all_scores=True
            )
            logger.info("✅ Sentiment model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment model failed, using fallback: {str(e)}")
            self.pipelines["sentiment"] = None
    
    async def _initialize_mental_health_model(self):
        """Initialize mental health classification model"""
        try:
            config = self.model_configs["mental_health"]
            
            # Try to load the specialized model
            try:
                self.tokenizers["mental_health"] = AutoTokenizer.from_pretrained(config["model_name"])
                self.models["mental_health"] = AutoModelForSequenceClassification.from_pretrained(config["model_name"])
                logger.info("✅ Mental health model loaded")
            except:
                # Fallback to general classification
                logger.warning("⚠️ Specialized mental health model not available, using general classification")
                self.pipelines["mental_health"] = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
        except Exception as e:
            logger.warning(f"⚠️ Mental health model initialization failed: {str(e)}")
            self.pipelines["mental_health"] = None
    
    async def _initialize_emotion_model(self):
        """Initialize emotion detection model"""
        try:
            config = self.model_configs["emotion"]
            self.pipelines["emotion"] = pipeline(
                config["task"],
                model=config["model_name"],
                return_all_scores=True
            )
            logger.info("✅ Emotion model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Emotion model failed: {str(e)}")
            self.pipelines["emotion"] = None
    
    async def _initialize_crisis_model(self):
        """Initialize crisis detection model"""
        try:
            # For crisis detection, we'll use a combination approach
            self.pipelines["crisis"] = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True
            )
            logger.info("✅ Crisis detection model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Crisis model failed: {str(e)}")
            self.pipelines["crisis"] = None
    
    async def _initialize_embeddings_model(self):
        """Initialize sentence embeddings model"""
        try:
            self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
            logger.info("✅ Embeddings model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Embeddings model failed: {str(e)}")
            self.embeddings_model = None
    
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings for texts"""
        if self.embeddings_model is None:
            raise ValueError("Embeddings model not available")
        
        return self.embeddings_model.encode(texts)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.embeddings_model is None:
            return 0.0
        
        embeddings = self.get_text_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

# Global model manager instance
ai_models = AIModelManager()
