"""
Evaluator module for sentiment analysis models.

This module provides the Evaluator class for calculating performance metrics,
confusion matrices, comparison tables, and error analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class Evaluator:
    """Evaluates and compares classifiers."""
    
    def __init__(self):
        """Initialize the evaluator with empty result storage."""
        self.results: Dict[str, Dict[str, float]] = {}
        self.confusion_matrices: Dict[str, np.ndarray] = {}
        self.training_times: Dict[str, float] = {}
        self.inference_times: Dict[str, float] = {}
    
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluates predictions and stores results.
        
        Calculates accuracy, precision (macro and per-class), recall (macro and per-class),
        F1-score (macro and weighted), and confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with all calculated metrics
        """
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_class_0': precision_score(y_true, y_pred, average=None, zero_division=0)[0],
            'precision_class_1': precision_score(y_true, y_pred, average=None, zero_division=0)[1],
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_class_0': recall_score(y_true, y_pred, average=None, zero_division=0)[0],
            'recall_class_1': recall_score(y_true, y_pred, average=None, zero_division=0)[1],
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Store results
        self.results[model_name] = metrics
        
        # Calculate and store confusion matrix
        cm = self.get_confusion_matrix(y_true, y_pred)
        self.confusion_matrices[model_name] = cm
        
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculates confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as numpy array
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Returns comparison table of all models with mean ± std.
        
        This method is designed to work with results from multiple simulations.
        For single evaluations, it returns the raw metrics without std.
        
        Returns:
            DataFrame with model comparison (columns: model names, rows: metrics)
        """
        if not self.results:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results).T
        
        # Select key metrics for comparison
        key_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        comparison_df = df[available_metrics].copy()
        
        # Add timing information if available
        if self.training_times:
            comparison_df['training_time'] = pd.Series(self.training_times)
        if self.inference_times:
            comparison_df['inference_time'] = pd.Series(self.inference_times)
        
        return comparison_df
    
    def get_error_examples(
        self, 
        texts: List[str], 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        n_examples: int = 5
    ) -> pd.DataFrame:
        """
        Returns error examples for qualitative analysis.
        
        Identifies misclassified examples and returns them for manual inspection.
        
        Args:
            texts: List of text samples
            y_true: True labels
            y_pred: Predicted labels
            n_examples: Number of error examples to return (default 5)
            
        Returns:
            DataFrame with error examples containing text, true label, and predicted label
        """
        # Find misclassified indices
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        
        if len(error_indices) == 0:
            return pd.DataFrame(columns=['text', 'true_label', 'predicted_label'])
        
        # Sample up to n_examples
        sample_size = min(n_examples, len(error_indices))
        sampled_indices = np.random.choice(error_indices, size=sample_size, replace=False)
        
        # Create DataFrame with error examples
        error_examples = pd.DataFrame({
            'text': [texts[i] for i in sampled_indices],
            'true_label': y_true[sampled_indices],
            'predicted_label': y_pred[sampled_indices]
        })
        
        return error_examples
    
    def add_timing(self, model_name: str, training_time: float = None, inference_time: float = None):
        """
        Adds timing information for a model.
        
        Args:
            model_name: Name of the model
            training_time: Training time in seconds (optional)
            inference_time: Inference time in seconds (optional)
        """
        if training_time is not None:
            self.training_times[model_name] = training_time
        if inference_time is not None:
            self.inference_times[model_name] = inference_time
    
    def get_summary(self, model_name: str = None) -> str:
        """
        Returns a formatted summary of evaluation results.
        
        Args:
            model_name: Specific model to summarize (if None, summarizes all models)
            
        Returns:
            Formatted string with evaluation summary
        """
        if model_name:
            if model_name not in self.results:
                return f"No results found for model: {model_name}"
            
            metrics = self.results[model_name]
            summary = f"\n=== {model_name} ===\n"
            summary += f"Accuracy: {metrics['accuracy']:.4f}\n"
            summary += f"F1-Score (macro): {metrics['f1_macro']:.4f}\n"
            summary += f"F1-Score (weighted): {metrics['f1_weighted']:.4f}\n"
            summary += f"Precision (macro): {metrics['precision_macro']:.4f}\n"
            summary += f"Recall (macro): {metrics['recall_macro']:.4f}\n"
            
            if model_name in self.training_times:
                summary += f"Training time: {self.training_times[model_name]:.2f}s\n"
            if model_name in self.inference_times:
                summary += f"Inference time: {self.inference_times[model_name]:.2f}s\n"
            
            return summary
        else:
            # Summarize all models
            summaries = []
            for name in self.results.keys():
                summaries.append(self.get_summary(name))
            return "\n".join(summaries)
