"""
Integration tests for SVM classifier with TF-IDF vectors.

These tests verify that the SVMClassifier works correctly with
TF-IDF vectors as specified in Requirement 3.4.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.classifiers import SVMClassifier


class TestSVMClassifierWithTFIDF:
    """Test SVMClassifier with TF-IDF-like sparse matrices."""
    
    def test_linear_svm_with_sparse_tfidf(self):
        """Test linear SVM with sparse TF-IDF vectors (Requirement 3.4)."""
        # Create sparse TF-IDF-like data
        # Simulating TF-IDF vectors with 5000 features
        np.random.seed(42)
        n_samples = 100
        n_features = 5000
        
        # Create sparse data (TF-IDF is typically sparse)
        X_train_dense = np.random.rand(n_samples, n_features)
        X_train_dense[X_train_dense < 0.95] = 0  # Make it sparse
        X_train = csr_matrix(X_train_dense)
        
        # Binary labels (positive/negative sentiment)
        y_train = np.random.randint(0, 2, n_samples)
        
        # Initialize with linear kernel and C=1.0 as per requirement
        clf = SVMClassifier(kernel='linear', C=1.0)
        
        # Train the model
        clf.fit(X_train, y_train)
        
        # Verify model is trained
        assert hasattr(clf.model, 'classes_')
        assert len(clf.model.classes_) == 2
        
        # Test prediction
        X_test_dense = np.random.rand(10, n_features)
        X_test_dense[X_test_dense < 0.95] = 0
        X_test = csr_matrix(X_test_dense)
        
        predictions = clf.predict(X_test)
        
        # Verify predictions
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_linear_svm_default_parameters(self):
        """Test that default parameters match requirement 3.4 (linear kernel, C=1.0)."""
        clf = SVMClassifier()  # Should default to linear kernel
        
        assert clf.kernel == 'linear'
        assert clf.C == 1.0
    
    def test_linear_svm_with_realistic_sentiment_data(self):
        """Test with more realistic sentiment classification scenario."""
        np.random.seed(42)
        
        # Simulate 200 reviews with 5000 TF-IDF features
        n_train = 200
        n_test = 50
        n_features = 5000
        
        # Create training data
        # Positive reviews (class 1) tend to have higher values in certain features
        X_pos = np.random.rand(n_train // 2, n_features)
        X_pos[:, :100] *= 2  # Boost "positive" features
        X_pos[X_pos < 0.9] = 0
        
        # Negative reviews (class 0) tend to have higher values in different features
        X_neg = np.random.rand(n_train // 2, n_features)
        X_neg[:, 100:200] *= 2  # Boost "negative" features
        X_neg[X_neg < 0.9] = 0
        
        X_train = csr_matrix(np.vstack([X_pos, X_neg]))
        y_train = np.array([1] * (n_train // 2) + [0] * (n_train // 2))
        
        # Shuffle
        shuffle_idx = np.random.permutation(n_train)
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        # Train classifier with requirement 3.4 parameters
        clf = SVMClassifier(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)
        
        # Create test data with similar pattern
        X_pos_test = np.random.rand(n_test // 2, n_features)
        X_pos_test[:, :100] *= 2
        X_pos_test[X_pos_test < 0.9] = 0
        
        X_neg_test = np.random.rand(n_test // 2, n_features)
        X_neg_test[:, 100:200] *= 2
        X_neg_test[X_neg_test < 0.9] = 0
        
        X_test = csr_matrix(np.vstack([X_pos_test, X_neg_test]))
        y_test = np.array([1] * (n_test // 2) + [0] * (n_test // 2))
        
        # Predict
        predictions = clf.predict(X_test)
        
        # Verify predictions
        assert len(predictions) == n_test
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should achieve reasonable accuracy on this synthetic data
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.5  # Better than random
    
    def test_handles_dense_arrays(self):
        """Test that classifier also works with dense arrays (not just sparse)."""
        clf = SVMClassifier(kernel='linear', C=1.0)
        
        # Dense TF-IDF-like data
        X_train = np.random.rand(50, 100)
        y_train = np.random.randint(0, 2, 50)
        
        clf.fit(X_train, y_train)
        
        X_test = np.random.rand(10, 100)
        predictions = clf.predict(X_test)
        
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
