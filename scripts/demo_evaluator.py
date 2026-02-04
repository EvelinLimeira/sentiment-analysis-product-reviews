"""
Demonstration script for the Evaluator module.

This script shows how to use the Evaluator class to evaluate and compare
sentiment analysis models.
"""

import numpy as np
from src.evaluator import Evaluator


def main():
    """Demonstrate evaluator functionality."""
    print("=" * 60)
    print("Evaluator Module Demonstration")
    print("=" * 60)
    
    # Create evaluator instance
    evaluator = Evaluator()
    
    # Simulate test data (100 samples, balanced)
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)
    
    # Simulate predictions from different models
    # Model 1: SVM with BoW (85% accuracy)
    y_pred_svm_bow = y_true.copy()
    errors_svm = np.random.choice(100, size=15, replace=False)
    y_pred_svm_bow[errors_svm] = 1 - y_pred_svm_bow[errors_svm]
    
    # Model 2: SVM with Embeddings (88% accuracy)
    y_pred_svm_emb = y_true.copy()
    errors_emb = np.random.choice(100, size=12, replace=False)
    y_pred_svm_emb[errors_emb] = 1 - y_pred_svm_emb[errors_emb]
    
    # Model 3: BERT (92% accuracy)
    y_pred_bert = y_true.copy()
    errors_bert = np.random.choice(100, size=8, replace=False)
    y_pred_bert[errors_bert] = 1 - y_pred_bert[errors_bert]
    
    print("\n1. Evaluating Models")
    print("-" * 60)
    
    # Evaluate all models
    print("Evaluating SVM + BoW...")
    evaluator.evaluate(y_true, y_pred_svm_bow, "SVM-BoW")
    evaluator.add_timing("SVM-BoW", training_time=5.2, inference_time=0.3)
    
    print("Evaluating SVM + Embeddings...")
    evaluator.evaluate(y_true, y_pred_svm_emb, "SVM-Embeddings")
    evaluator.add_timing("SVM-Embeddings", training_time=8.7, inference_time=0.5)
    
    print("Evaluating BERT...")
    evaluator.evaluate(y_true, y_pred_bert, "BERT")
    evaluator.add_timing("BERT", training_time=120.5, inference_time=2.1)
    
    print("\n2. Model Summaries")
    print("-" * 60)
    print(evaluator.get_summary())
    
    print("\n3. Comparison Table")
    print("-" * 60)
    comparison = evaluator.get_comparison_table()
    print(comparison.round(4))
    
    print("\n4. Confusion Matrices")
    print("-" * 60)
    for model_name, cm in evaluator.confusion_matrices.items():
        print(f"\n{model_name}:")
        print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
        print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    print("\n5. Error Examples (SVM-BoW)")
    print("-" * 60)
    # Create sample texts for error analysis
    texts = [f"Review {i}: Sample text about product" for i in range(100)]
    errors = evaluator.get_error_examples(texts, y_true, y_pred_svm_bow, n_examples=3)
    print(errors.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
