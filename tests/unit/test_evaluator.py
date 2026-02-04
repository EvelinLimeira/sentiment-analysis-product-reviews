"""
Unit tests for the Evaluator class.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluator import Evaluator


class TestEvaluator:
    """Test suite for Evaluator class."""
    
    def test_evaluator_initialization(self):
        """Test that evaluator initializes with empty storage."""
        evaluator = Evaluator()
        assert evaluator.results == {}
        assert evaluator.confusion_matrices == {}
        assert evaluator.training_times == {}
        assert evaluator.inference_times == {}
    
    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        metrics = evaluator.evaluate(y_true, y_pred, "perfect_model")
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0
        assert metrics['f1_macro'] == 1.0
        assert metrics['f1_weighted'] == 1.0
        assert "perfect_model" in evaluator.results
        assert "perfect_model" in evaluator.confusion_matrices
    
    def test_evaluate_imperfect_predictions(self):
        """Test evaluation with some errors."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])  # 2 errors
        
        metrics = evaluator.evaluate(y_true, y_pred, "imperfect_model")
        
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['accuracy'] == 0.75  # 6/8 correct
        assert 0.0 <= metrics['f1_macro'] <= 1.0
        assert "imperfect_model" in evaluator.results
    
    def test_get_confusion_matrix(self):
        """Test confusion matrix calculation."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm = evaluator.get_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
        assert np.all(cm >= 0)
        # Check specific values
        assert cm[0, 0] == 2  # True negatives
        assert cm[0, 1] == 1  # False positives
        assert cm[1, 0] == 1  # False negatives
        assert cm[1, 1] == 2  # True positives
    
    def test_get_comparison_table_empty(self):
        """Test comparison table with no results."""
        evaluator = Evaluator()
        df = evaluator.get_comparison_table()
        assert df.empty
    
    def test_get_comparison_table_single_model(self):
        """Test comparison table with one model."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        evaluator.evaluate(y_true, y_pred, "model1")
        df = evaluator.get_comparison_table()
        
        assert not df.empty
        assert "model1" in df.index
        assert "accuracy" in df.columns
        assert "f1_macro" in df.columns
    
    def test_get_comparison_table_multiple_models(self):
        """Test comparison table with multiple models."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1])
        
        evaluator.evaluate(y_true, np.array([0, 0, 1, 1]), "model1")
        evaluator.evaluate(y_true, np.array([0, 1, 1, 0]), "model2")
        
        df = evaluator.get_comparison_table()
        
        assert len(df) == 2
        assert "model1" in df.index
        assert "model2" in df.index
    
    def test_get_error_examples_no_errors(self):
        """Test error examples with perfect predictions."""
        evaluator = Evaluator()
        texts = ["text1", "text2", "text3", "text4"]
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        errors = evaluator.get_error_examples(texts, y_true, y_pred, n_examples=5)
        
        assert len(errors) == 0
        assert "text" in errors.columns
        assert "true_label" in errors.columns
        assert "predicted_label" in errors.columns
    
    def test_get_error_examples_with_errors(self):
        """Test error examples with misclassifications."""
        evaluator = Evaluator()
        texts = ["text1", "text2", "text3", "text4"]
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])  # 2 errors
        
        errors = evaluator.get_error_examples(texts, y_true, y_pred, n_examples=5)
        
        assert len(errors) == 2  # Should return 2 errors
        assert all(errors['true_label'] != errors['predicted_label'])
        assert "text2" in errors['text'].values or "text4" in errors['text'].values
    
    def test_get_error_examples_limit(self):
        """Test that error examples respects n_examples limit."""
        evaluator = Evaluator()
        texts = [f"text{i}" for i in range(10)]
        y_true = np.array([0] * 10)
        y_pred = np.array([1] * 10)  # All wrong
        
        errors = evaluator.get_error_examples(texts, y_true, y_pred, n_examples=3)
        
        assert len(errors) == 3
    
    def test_add_timing(self):
        """Test adding timing information."""
        evaluator = Evaluator()
        
        evaluator.add_timing("model1", training_time=10.5, inference_time=2.3)
        
        assert evaluator.training_times["model1"] == 10.5
        assert evaluator.inference_times["model1"] == 2.3
    
    def test_add_timing_partial(self):
        """Test adding only training or inference time."""
        evaluator = Evaluator()
        
        evaluator.add_timing("model1", training_time=10.5)
        evaluator.add_timing("model2", inference_time=2.3)
        
        assert "model1" in evaluator.training_times
        assert "model1" not in evaluator.inference_times
        assert "model2" in evaluator.inference_times
        assert "model2" not in evaluator.training_times
    
    def test_get_summary_single_model(self):
        """Test summary generation for a single model."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        evaluator.evaluate(y_true, y_pred, "test_model")
        evaluator.add_timing("test_model", training_time=5.0, inference_time=1.0)
        
        summary = evaluator.get_summary("test_model")
        
        assert "test_model" in summary
        assert "Accuracy" in summary
        assert "F1-Score" in summary
        assert "Training time" in summary
        assert "Inference time" in summary
    
    def test_get_summary_all_models(self):
        """Test summary generation for all models."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1])
        
        evaluator.evaluate(y_true, np.array([0, 0, 1, 1]), "model1")
        evaluator.evaluate(y_true, np.array([0, 1, 1, 0]), "model2")
        
        summary = evaluator.get_summary()
        
        assert "model1" in summary
        assert "model2" in summary
    
    def test_get_summary_nonexistent_model(self):
        """Test summary for a model that doesn't exist."""
        evaluator = Evaluator()
        summary = evaluator.get_summary("nonexistent")
        
        assert "No results found" in summary
    
    def test_metrics_range_validity(self):
        """Test that all metrics are in valid range [0, 1]."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        
        metrics = evaluator.evaluate(y_true, y_pred, "test_model")
        
        # Check all metrics are in [0, 1]
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range [0, 1]"
    
    def test_confusion_matrix_validity(self):
        """Test that confusion matrix has valid properties."""
        evaluator = Evaluator()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        evaluator.evaluate(y_true, y_pred, "test_model")
        cm = evaluator.confusion_matrices["test_model"]
        
        # Sum should equal total samples
        assert cm.sum() == len(y_true)
        # All elements should be non-negative
        assert np.all(cm >= 0)
        # All elements should be integers
        assert cm.dtype in [np.int32, np.int64]
