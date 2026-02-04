"""
Integration tests for the Evaluator with actual models.
"""

import pytest
import numpy as np
from src.evaluator import Evaluator


class TestEvaluatorIntegration:
    """Integration tests for Evaluator with realistic scenarios."""
    
    def test_evaluator_with_multiple_models(self):
        """Test evaluator comparing multiple models."""
        evaluator = Evaluator()
        
        # Simulate test data
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        # Model 1: Good performance
        y_pred_model1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])  # 80% accuracy
        
        # Model 2: Better performance
        y_pred_model2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 100% accuracy
        
        # Model 3: Poor performance
        y_pred_model3 = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])  # 40% accuracy
        
        # Evaluate all models
        metrics1 = evaluator.evaluate(y_true, y_pred_model1, "SVM-BoW")
        metrics2 = evaluator.evaluate(y_true, y_pred_model2, "BERT")
        metrics3 = evaluator.evaluate(y_true, y_pred_model3, "Baseline")
        
        # Add timing information
        evaluator.add_timing("SVM-BoW", training_time=5.2, inference_time=0.3)
        evaluator.add_timing("BERT", training_time=120.5, inference_time=2.1)
        evaluator.add_timing("Baseline", training_time=0.1, inference_time=0.05)
        
        # Verify results are stored
        assert len(evaluator.results) == 3
        assert "SVM-BoW" in evaluator.results
        assert "BERT" in evaluator.results
        assert "Baseline" in evaluator.results
        
        # Verify BERT has best accuracy
        assert metrics2['accuracy'] > metrics1['accuracy']
        assert metrics2['accuracy'] > metrics3['accuracy']
        
        # Get comparison table
        comparison = evaluator.get_comparison_table()
        assert len(comparison) == 3
        assert "accuracy" in comparison.columns
        assert "training_time" in comparison.columns
        
        # Verify BERT is best in comparison table
        assert comparison.loc["BERT", "accuracy"] == 1.0
    
    def test_evaluator_error_analysis(self):
        """Test error analysis functionality."""
        evaluator = Evaluator()
        
        # Create test data with known errors
        texts = [
            "This product is terrible",  # True: 0, Pred: 0 ✓
            "Worst purchase ever",       # True: 0, Pred: 1 ✗
            "Amazing quality!",          # True: 1, Pred: 1 ✓
            "Love it so much",           # True: 1, Pred: 0 ✗
            "Not recommended",           # True: 0, Pred: 0 ✓
        ]
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0])
        
        # Evaluate
        evaluator.evaluate(y_true, y_pred, "test_model")
        
        # Get error examples
        errors = evaluator.get_error_examples(texts, y_true, y_pred, n_examples=5)
        
        # Should have exactly 2 errors
        assert len(errors) == 2
        
        # Verify all returned examples are actual errors
        for _, row in errors.iterrows():
            assert row['true_label'] != row['predicted_label']
        
        # Verify the error texts are correct
        error_texts = set(errors['text'].values)
        assert "Worst purchase ever" in error_texts or "Love it so much" in error_texts
    
    def test_evaluator_confusion_matrix_interpretation(self):
        """Test confusion matrix with clear interpretation."""
        evaluator = Evaluator()
        
        # Create balanced test set
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Model predicts: 3 TN, 1 FP, 1 FN, 3 TP
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        
        evaluator.evaluate(y_true, y_pred, "test_model")
        cm = evaluator.confusion_matrices["test_model"]
        
        # Verify confusion matrix structure
        # [[TN, FP],
        #  [FN, TP]]
        assert cm[0, 0] == 3  # True Negatives
        assert cm[0, 1] == 1  # False Positives
        assert cm[1, 0] == 1  # False Negatives
        assert cm[1, 1] == 3  # True Positives
        
        # Verify metrics match confusion matrix
        metrics = evaluator.results["test_model"]
        expected_accuracy = (3 + 3) / 8  # (TN + TP) / Total
        assert abs(metrics['accuracy'] - expected_accuracy) < 0.001
    
    def test_evaluator_summary_output(self):
        """Test that summary provides useful information."""
        evaluator = Evaluator()
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        evaluator.evaluate(y_true, y_pred, "Perfect Model")
        evaluator.add_timing("Perfect Model", training_time=10.0, inference_time=1.0)
        
        summary = evaluator.get_summary("Perfect Model")
        
        # Verify summary contains key information
        assert "Perfect Model" in summary
        assert "1.0000" in summary  # Perfect accuracy
        assert "10.00s" in summary  # Training time
        assert "1.00s" in summary   # Inference time
    
    def test_evaluator_with_imbalanced_data(self):
        """Test evaluator with imbalanced dataset."""
        evaluator = Evaluator()
        
        # Highly imbalanced: 90% negative, 10% positive
        y_true = np.array([0] * 90 + [1] * 10)
        # Naive model that always predicts negative
        y_pred = np.array([0] * 100)
        
        metrics = evaluator.evaluate(y_true, y_pred, "Naive Model")
        
        # High accuracy due to imbalance
        assert metrics['accuracy'] == 0.9
        
        # But poor recall for positive class
        assert metrics['recall_class_1'] == 0.0
        
        # F1 should reflect the imbalance
        assert metrics['f1_macro'] < metrics['accuracy']
