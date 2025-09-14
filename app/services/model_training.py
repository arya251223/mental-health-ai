import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime
from loguru import logger

class MentalHealthDataset(Dataset):
    """Custom dataset for mental health text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MentalHealthModelTrainer:
    """Train and fine-tune models for mental health classification"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Label mappings for different tasks
        self.label_mappings = {
            "risk_assessment": {
                "low": 0, "medium": 1, "high": 2, "critical": 3
            },
            "crisis_detection": {
                "no_crisis": 0, "crisis": 1
            },
            "condition_classification": {
                "depression": 0, "anxiety": 1, "bipolar": 2, 
                "adhd": 3, "ocd": 4, "ptsd": 5, "normal": 6
            }
        }
    
    def prepare_data(
        self, 
        texts: List[str], 
        labels: List[str], 
        task_type: str = "risk_assessment"
    ) -> Tuple[List[str], List[int]]:
        """Prepare and clean data for training"""
        
        # Clean texts
        cleaned_texts = []
        for text in texts:
            # Basic text cleaning
            text = text.strip().lower()
            text = ' '.join(text.split())  # Remove extra whitespace
            cleaned_texts.append(text)
        
        # Convert labels to integers
        label_map = self.label_mappings.get(task_type, {})
        numeric_labels = []
        
        for label in labels:
            if label in label_map:
                numeric_labels.append(label_map[label])
            else:
                logger.warning(f"Unknown label: {label}, skipping...")
                continue
        
        # Ensure we have matching text and label counts
        min_length = min(len(cleaned_texts), len(numeric_labels))
        cleaned_texts = cleaned_texts[:min_length]
        numeric_labels = numeric_labels[:min_length]
        
        logger.info(f"Prepared {len(cleaned_texts)} samples for {task_type}")
        return cleaned_texts, numeric_labels
    
    def initialize_model(self, num_labels: int, task_type: str = "risk_assessment"):
        """Initialize tokenizer and model"""
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                problem_type="single_label_classification"
            )
            
            # Add special tokens if needed
            special_tokens = {
                "additional_special_tokens": [
                    "[CRISIS]", "[DEPRESSION]", "[ANXIETY]", "[NORMAL]"
                ]
            }
            
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Model initialized for {task_type} with {num_labels} labels")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def train_model(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "./models",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> Dict[str, Any]:
        """Train the mental health classification model"""
        
        try:
            # Create datasets
            train_dataset = MentalHealthDataset(
                train_texts, train_labels, self.tokenizer
            )
            
            val_dataset = None
            if val_texts and val_labels:
                val_dataset = MentalHealthDataset(
                    val_texts, val_labels, self.tokenizer
                )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=100,
                evaluation_strategy="epoch" if val_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_accuracy" if val_dataset else None,
                learning_rate=learning_rate,
                seed=42
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train model
            logger.info("Starting model training...")
            train_result = self.trainer.train()
            
            # Save model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Evaluate model
            if val_dataset:
                eval_result = self.trainer.evaluate()
                logger.info(f"Validation results: {eval_result}")
                return {
                    "train_result": train_result,
                    "eval_result": eval_result
                }
            
            logger.info("Model training completed successfully")
            return {"train_result": train_result}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def create_synthetic_training_data(self, task_type: str = "risk_assessment") -> Tuple[List[str], List[str]]:
        """Create synthetic training data for demonstration"""
        
        synthetic_data = {
            "risk_assessment": {
                "low": [
                    "I'm feeling pretty good today, just checking in",
                    "Had a nice day with friends, feeling grateful",
                    "Work was stressful but I managed okay",
                    "Feeling optimistic about the future",
                    "Just dealing with normal life challenges"
                ],
                "medium": [
                    "I've been feeling down lately but it's manageable",
                    "Anxiety is bothering me more than usual",
                    "Having trouble sleeping and concentrating",
                    "Feeling overwhelmed with everything going on",
                    "Not sure how to cope with these feelings"
                ],
                "high": [
                    "I can't handle this anymore, everything feels hopeless",
                    "Having really dark thoughts and can't shake them",
                    "Feel like I'm completely alone in this world",
                    "Nothing seems to help, I'm losing hope",
                    "Can't see any way things will get better"
                ],
                "critical": [
                    "I want to end my life, can't take this pain anymore",
                    "Thinking about suicide constantly, have a plan",
                    "Feel like everyone would be better off without me",
                    "Going to hurt myself tonight, can't go on",
                    "Life isn't worth living anymore, ready to die"
                ]
            },
            "crisis_detection": {
                "no_crisis": [
                    "Feeling sad but I have good support",
                    "Anxious about work but managing with therapy",
                    "Going through a tough time but staying strong",
                    "Stressed but using healthy coping mechanisms",
                    "Down but know this will pass"
                ],
                "crisis": [
                    "Want to kill myself, have pills ready",
                    "Going to jump from the bridge tonight",
                    "Can't live anymore, planning to die",
                    "Ready to end it all, goodbye world",
                    "Suicide is my only option now"
                ]
            }
        }
        
        if task_type not in synthetic_data:
            raise ValueError(f"Unknown task type: {task_type}")
        
        texts = []
        labels = []
        
        for label, examples in synthetic_data[task_type].items():
            texts.extend(examples * 10)  # Multiply for more training data
            labels.extend([label] * len(examples) * 10)
        
        logger.info(f"Generated {len(texts)} synthetic samples for {task_type}")
        return texts, labels

