import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from loguru import logger
import joblib
from datetime import datetime

class MentalHealthModelInference:
    """Handle model inference for mental health predictions"""
    
    def __init__(self, models_dir: str = "./data/models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.tokenizers = {}
        
        # Reverse label mappings for prediction interpretation
        self.reverse_label_mappings = {
            "risk_assessment": {
                0: "low", 1: "medium", 2: "high", 3: "critical"
            },
            "crisis_detection": {
                0: "no_crisis", 1: "crisis"
            },
            "condition_classification": {
                0: "depression", 1: "anxiety", 2: "bipolar",
                3: "adhd", 4: "ocd", 5: "ptsd", 6: "normal"
            }
        }
    
    def load_model(self, model_type: str, model_path: Optional[str] = None) -> bool:
        """Load a trained model for inference"""
        
        try:
            if model_path is None:
                model_path = os.path.join(self.models_dir, model_type)
            
            if not os.path.exists(model_path):
                logger.warning(f"Model path not found: {model_path}")
                return False
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Store loaded components
            self.tokenizers[model_type] = tokenizer
            self.loaded_models[model_type] = model
            
            logger.info(f"Successfully loaded {model_type} model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {str(e)}")
            return False
    
    def predict_risk_level(self, text: str, model_type: str = "risk_assessment") -> Dict[str, Any]:
        """Predict risk level from text"""
        
        if model_type not in self.loaded_models:
            logger.warning(f"Model {model_type} not loaded, using fallback")
            return self._fallback_risk_prediction(text)
        
        try:
            tokenizer = self.tokenizers[model_type]
            model = self.loaded_models[model_type]
            
            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
            
            # Convert to probabilities
            probabilities = predictions.numpy()[0]
            labels = self.reverse_label_mappings[model_type]
            
            # Create result
            result = {}
            predicted_class = np.argmax(probabilities)
            
            for idx, prob in enumerate(probabilities):
                label = labels.get(idx, f"class_{idx}")
                result[label] = float(prob)
            
            result["predicted_risk"] = labels[predicted_class]
            result["confidence"] = float(probabilities[predicted_class])
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._fallback_risk_prediction(text)
    
    def predict_crisis(self, text: str) -> Dict[str, Any]:
        """Predict crisis situation from text"""
        
        # Use pattern-based detection as primary method
        crisis_patterns = [
            r"(?:want to|going to|plan to) (?:die|kill myself|end (?:it|my life))",
            r"suicide (?:plan|method)",
            r"(?:everyone|world) (?:better|off) without me",
            r"can't (?:take|handle|go on) anymore"
        ]
        
        text_lower = text.lower()
        crisis_score = 0
        detected_patterns = []
        
        for pattern in crisis_patterns:
            if re.search(pattern, text_lower):
                crisis_score += 1
                detected_patterns.append(pattern)
        
        # Crisis keywords
        crisis_keywords = ["suicide", "kill myself", "end it all", "want to die", "hopeless"]
        keyword_matches = [keyword for keyword in crisis_keywords if keyword in text_lower]
        
        crisis_score += len(keyword_matches)
        
        # Determine crisis level
        is_crisis = crisis_score >= 2
        severity = min(crisis_score * 2, 10)
        
        return {
            "is_crisis": is_crisis,
            "severity_level": severity,
            "crisis_indicators": detected_patterns + keyword_matches,
            "confidence": min(crisis_score / 5.0, 1.0),
            "timestamp": datetime.now().isoformat()
        }
    
    def predict_conditions(self, text: str) -> Dict[str, float]:
        """Predict mental health conditions from text"""
        
        condition_keywords = {
            "depression": ["depressed", "sad", "hopeless", "worthless", "empty", "crying"],
            "anxiety": ["anxious", "worried", "panic", "nervous", "overwhelmed", "stress"],
            "bipolar": ["manic", "mood swings", "high energy", "ups and downs"],
            "adhd": ["can't focus", "distracted", "hyperactive", "impulsive"],
            "ocd": ["obsessive", "compulsive", "repetitive", "checking", "rituals"],
            "ptsd": ["trauma", "flashbacks", "nightmares", "triggers", "hypervigilant"]
        }
        
        text_lower = text.lower()
        word_count = len(text_lower.split())
        
        condition_scores = {}
        
        for condition, keywords in condition_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Normalize by text length and keyword count
                score = (matches / len(keywords)) * (matches / max(word_count / 10, 1))
                condition_scores[condition] = min(score, 1.0)
        
        return condition_scores
    
    def _fallback_risk_prediction(self, text: str) -> Dict[str, Any]:
        """Fallback risk prediction using simple heuristics"""
        
        text_lower = text.lower()
        
        # High-risk indicators
        high_risk_words = ["suicide", "kill myself", "want to die", "end it all"]
        high_risk_score = sum(1 for word in high_risk_words if word in text_lower)
        
        # Medium-risk indicators
        medium_risk_words = ["hopeless", "can't handle", "overwhelming", "depressed"]
        medium_risk_score = sum(1 for word in medium_risk_words if word in text_lower)
        
        # Low-risk indicators
        positive_words = ["good", "better", "hopeful", "grateful", "happy"]
        positive_score = sum(1 for word in positive_words if word in text_lower)
        
        # Calculate risk level
        if high_risk_score >= 2:
            risk = "critical"
            confidence = 0.9
        elif high_risk_score >= 1:
            risk = "high" 
            confidence = 0.8
        elif medium_risk_score >= 2:
            risk = "medium"
            confidence = 0.7
        elif positive_score > medium_risk_score:
            risk = "low"
            confidence = 0.6
        else:
            risk = "medium"
            confidence = 0.5
        
        return {
            "low": 0.1 if risk != "low" else 0.7,
            "medium": 0.6 if risk == "medium" else 0.2,
            "high": 0.8 if risk == "high" else 0.1,
            "critical": 0.9 if risk == "critical" else 0.05,
            "predicted_risk": risk,
            "confidence": confidence,
            "method": "fallback_heuristic",
            "timestamp": datetime.now().isoformat()
        }

# Global inference instance
model_inference = MentalHealthModelInference()

