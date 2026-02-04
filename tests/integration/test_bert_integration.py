"""
Integration test for BERT classifier.

This test demonstrates the complete workflow:
1. Load DistilBERT model
2. Tokenize texts with truncation and padding
3. Fine-tune on training data with validation
4. Predict on test data
"""

import pytest
import numpy as np
from src.bert_classifier import BERTClassifier


@pytest.mark.integration
def test_bert_complete_workflow():
    """Test complete BERT workflow from training to prediction."""
    
    # Sample training data
    train_texts = [
        "This product is absolutely amazing! Best purchase ever.",
        "Excellent quality and fast shipping. Highly recommend!",
        "Love it! Works perfectly and looks great.",
        "Outstanding product, exceeded my expectations.",
        "Terrible product, complete waste of money.",
        "Very disappointed with the quality. Do not buy.",
        "Broke after one day. Worst purchase ever.",
        "Poor quality and bad customer service."
    ]
    train_labels = [1, 1, 1, 1, 0, 0, 0, 0]
    
    # Sample validation data
    val_texts = [
        "Great product, very satisfied!",
        "Not worth the money, very disappointed."
    ]
    val_labels = [1, 0]
    
    # Sample test data
    test_texts = [
        "This is fantastic!",
        "This is horrible."
    ]
    expected_labels = [1, 0]
    
    # Initialize classifier with small batch size and 1 epoch for speed
    classifier = BERTClassifier(
        model_name='distilbert-base-uncased',
        max_length=128,
        batch_size=4,
        learning_rate=2e-5,
        num_epochs=1,
        patience=1
    )
    
    # Verify tokenization works
    encodings = classifier.tokenize(train_texts)
    assert 'input_ids' in encodings
    assert 'attention_mask' in encodings
    assert encodings['input_ids'].shape[0] == len(train_texts)
    assert encodings['input_ids'].shape[1] <= 128  # Respects max_length
    
    # Train the model
    classifier.fit(train_texts, train_labels, val_texts, val_labels)
    
    # Verify model is fitted
    assert classifier.is_fitted is True
    assert classifier.model is not None
    
    # Make predictions
    predictions = classifier.predict(test_texts)
    
    # Verify predictions
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_texts)
    assert all(pred in [0, 1] for pred in predictions)
    
    # Get probabilities
    probabilities = classifier.predict_proba(test_texts)
    
    # Verify probabilities
    assert probabilities.shape == (len(test_texts), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    
    print(f"\nTest Results:")
    print(f"Predictions: {predictions}")
    print(f"Expected: {expected_labels}")
    print(f"Probabilities:\n{probabilities}")


@pytest.mark.integration
def test_bert_handles_long_texts():
    """Test that BERT properly truncates long texts."""
    
    # Create very long texts
    long_positive = " ".join(["This is an amazing product"] * 100)
    long_negative = " ".join(["This is a terrible product"] * 100)
    
    train_texts = [long_positive, long_negative] * 4
    train_labels = [1, 0] * 4
    
    val_texts = [long_positive, long_negative]
    val_labels = [1, 0]
    
    classifier = BERTClassifier(
        max_length=128,
        batch_size=4,
        num_epochs=1
    )
    
    # Should handle long texts without error
    classifier.fit(train_texts, train_labels, val_texts, val_labels)
    
    # Test prediction on long text
    test_texts = [long_positive]
    predictions = classifier.predict(test_texts)
    
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]


@pytest.mark.integration
def test_bert_early_stopping():
    """Test that early stopping works correctly."""
    
    # Create simple data that should converge quickly
    train_texts = ["positive"] * 20 + ["negative"] * 20
    train_labels = [1] * 20 + [0] * 20
    
    val_texts = ["positive"] * 4 + ["negative"] * 4
    val_labels = [1] * 4 + [0] * 4
    
    classifier = BERTClassifier(
        batch_size=8,
        num_epochs=10,  # Set high
        patience=1  # Low patience
    )
    
    # Train with early stopping
    classifier.fit(train_texts, train_labels, val_texts, val_labels)
    
    # Should have best_model_state from early stopping
    assert hasattr(classifier, 'best_model_state')
    assert classifier.is_fitted is True


if __name__ == "__main__":
    # Run the integration test
    test_bert_complete_workflow()
    print("\n✓ Complete workflow test passed!")
    
    test_bert_handles_long_texts()
    print("✓ Long text handling test passed!")
    
    test_bert_early_stopping()
    print("✓ Early stopping test passed!")
