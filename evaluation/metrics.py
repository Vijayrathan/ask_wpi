#!/usr/bin/env python3
"""
Evaluation metrics for AskWPI models
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EvaluationMetrics:
    def __init__(self):
        """Initialize the evaluation metrics class"""
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for evaluation"""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        keywords = [word for word in tokens 
                   if word not in self.stop_words and len(word) > 2]
        
        return keywords
    
    def exact_match(self, predicted: str, ground_truth: str) -> float:
        """Calculate exact match score"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_clean = self.preprocess_text(predicted)
        ground_truth_clean = self.preprocess_text(ground_truth)
        
        return 1.0 if predicted_clean == ground_truth_clean else 0.0
    
    def semantic_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate semantic accuracy using sentence embeddings"""
        if not predicted or not ground_truth:
            return 0.0
        
        try:
            # Get embeddings
            pred_embedding = self.embedding_model.encode([predicted])[0]
            gt_embedding = self.embedding_model.encode([ground_truth])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(pred_embedding, gt_embedding) / (
                np.linalg.norm(pred_embedding) * np.linalg.norm(gt_embedding)
            )
            
            return float(similarity)
        except Exception as e:
            print(f"Error in semantic accuracy calculation: {e}")
            return 0.0
    
    def f1_score_custom(self, predicted: str, ground_truth: str) -> float:
        """Calculate F1 score based on keyword overlap"""
        if not predicted or not ground_truth:
            return 0.0
        
        pred_keywords = set(self.extract_keywords(predicted))
        gt_keywords = set(self.extract_keywords(ground_truth))
        
        if not gt_keywords:
            return 1.0 if not pred_keywords else 0.0
        
        # Calculate precision and recall
        if not pred_keywords:
            precision = 0.0
            recall = 0.0
        else:
            intersection = pred_keywords.intersection(gt_keywords)
            precision = len(intersection) / len(pred_keywords)
            recall = len(intersection) / len(gt_keywords)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def multi_turn_consistency(self, responses: List[str]) -> float:
        """Calculate consistency across multiple turns"""
        if len(responses) < 2:
            return 1.0
        
        # Calculate pairwise semantic similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if responses[i] and responses[j]:
                    sim = self.semantic_accuracy(responses[i], responses[j])
                    similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        return np.mean(similarities)
    
    def evaluate_batch(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate a batch of predictions"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        
        exact_matches = []
        semantic_accuracies = []
        f1_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            exact_matches.append(self.exact_match(pred, gt))
            semantic_accuracies.append(self.semantic_accuracy(pred, gt))
            f1_scores.append(self.f1_score_custom(pred, gt))
        
        return {
            'exact_match': np.mean(exact_matches),
            'semantic_accuracy': np.mean(semantic_accuracies),
            'f1_score': np.mean(f1_scores),
            'exact_match_std': np.std(exact_matches),
            'semantic_accuracy_std': np.std(semantic_accuracies),
            'f1_score_std': np.std(f1_scores)
        }
    
    def evaluate_multi_turn(self, conversation_responses: List[List[str]], 
                           ground_truths: List[List[str]]) -> Dict[str, float]:
        """Evaluate multi-turn conversations"""
        if len(conversation_responses) != len(ground_truths):
            raise ValueError("Conversation responses and ground truths must have the same length")
        
        consistency_scores = []
        overall_metrics = []
        
        for conv_responses, gt_responses in zip(conversation_responses, ground_truths):
            # Calculate consistency
            consistency = self.multi_turn_consistency(conv_responses)
            consistency_scores.append(consistency)
            
            # Calculate overall metrics for this conversation
            conv_metrics = self.evaluate_batch(conv_responses, gt_responses)
            overall_metrics.append(conv_metrics)
        
        # Aggregate results
        avg_consistency = np.mean(consistency_scores)
        avg_exact_match = np.mean([m['exact_match'] for m in overall_metrics])
        avg_semantic_accuracy = np.mean([m['semantic_accuracy'] for m in overall_metrics])
        avg_f1_score = np.mean([m['f1_score'] for m in overall_metrics])
        
        return {
            'multi_turn_consistency': avg_consistency,
            'exact_match': avg_exact_match,
            'semantic_accuracy': avg_semantic_accuracy,
            'f1_score': avg_f1_score,
            'consistency_std': np.std(consistency_scores)
        } 