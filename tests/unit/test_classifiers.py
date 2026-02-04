"""
Unit tests for SVM classifier module.
"""

import pytest
import numpy as np
from src.classifiers import SVMClassifier


class TestSVMClassifier:
    """Test suite for SVMClassifier class."""
    
    def test_init_linear_kernel(self):
        """Test initialization with linear kernel."""
        clf = SVMClassifier(kernel='linear', C=1.0)
        assert clf.kernel == 'linear'
        assert clf.C == 1.0
        assert clf.model is not None
    
    def test_init_rbf_kernel(self):
        """Test initialization with RBF kernel."""
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        assert clf.kernel == 'rbf'
        assert clf.C == 1.0
        assert clf.gamma == 'scale'
        assert clf.model is not None
    
    def test_init_invalid_kernel(self):
        """Test that invalid kernel raises ValueError."""
        with pytest.raises(ValueError, match="Kernel must be 'linear' or 'rbf'"):
            SVMClassifier(kernel='polynomial')
    
    def test_fit_basic(self):
        """Test basic fitting with simple data."""
        clf = SVMClassifier(kernel='linear')
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        
        result = clf.fit(X, y)
        assert result is clf  # Check method chaining
        assert hasattr(clf.model, 'classes_')
    
    def test_fit_empty_data(self):
        """Test that fitting with empty data raises ValueError."""
        clf = SVMClassifier(kernel='linear')
        X = np.array([])
        y = np.array([])
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            clf.fit(X, y)
    
    def test_fit_mismatched_shapes(self):
        """Test that mismatched X and y shapes raise ValueError."""
        clf = SVMClassifier(kernel='linear')
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 0, 1])  # Wrong length
        
        with pytest.raises(ValueError, match="X and y must have same length"):
            clf.fit(X, y)
    
    def test_predict_basic(self):
        """Test basic prediction after fitting."""
        clf = SVMClassifier(kernel='linear')
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 1, 1])
        
        clf.fit(X_train, y_train)
        
        X_test = np.array([[2, 3], [6, 7]])
        predictions = clf.predict(X_test)
        
        assert len(predictions) == 2
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_before_fit(self):
        """Test that predicting before fitting raises RuntimeError."""
        clf = SVMClassifier(kernel='linear')
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(RuntimeError, match="Model must be fitted before calling predict"):
            clf.predict(X)
    
    def test_predict_empty_data(self):
        """Test that predicting with empty data raises ValueError."""
        clf = SVMClassifier(kernel='linear')
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 1, 1])
        clf.fit(X_train, y_train)
        
        X_test = np.array([])
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            clf.predict(X_test)
    
    def test_linear_kernel_separable_data(self):
        """Test linear kernel on linearly separable data."""
        clf = SVMClassifier(kernel='linear', C=1.0)
        
        # Create linearly separable data
        np.random.seed(42)
        X_train = np.vstack([
            np.random.randn(20, 2) + [2, 2],  # Class 1
            np.random.randn(20, 2) + [-2, -2]  # Class 0
        ])
        y_train = np.array([1] * 20 + [0] * 20)
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_train)
        
        # Should achieve high accuracy on training data
        accuracy = np.mean(predictions == y_train)
        assert accuracy > 0.9
    
    def test_rbf_kernel_nonlinear_data(self):
        """Test RBF kernel on non-linearly separable data."""
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        
        # Create circular data (non-linearly separable)
        np.random.seed(42)
        n_samples = 100
        
        # Inner circle (class 0)
        r_inner = np.random.uniform(0, 1, n_samples // 2)
        theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)
        X_inner = np.column_stack([r_inner * np.cos(theta_inner), 
                                   r_inner * np.sin(theta_inner)])
        
        # Outer circle (class 1)
        r_outer = np.random.uniform(2, 3, n_samples // 2)
        theta_outer = np.random.uniform(0, 2 * np.pi, n_samples // 2)
        X_outer = np.column_stack([r_outer * np.cos(theta_outer), 
                                   r_outer * np.sin(theta_outer)])
        
        X_train = np.vstack([X_inner, X_outer])
        y_train = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_train)
        
        # RBF should handle this better than linear
        accuracy = np.mean(predictions == y_train)
        assert accuracy > 0.8
    
    def test_different_C_values(self):
        """Test that different C values affect the model."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        clf_low_c = SVMClassifier(kernel='linear', C=0.01)
        clf_high_c = SVMClassifier(kernel='linear', C=100.0)
        
        clf_low_c.fit(X, y)
        clf_high_c.fit(X, y)
        
        # Both should be able to fit
        assert hasattr(clf_low_c.model, 'classes_')
        assert hasattr(clf_high_c.model, 'classes_')
    
    def test_binary_classification(self):
        """Test that classifier works for binary classification."""
        clf = SVMClassifier(kernel='linear')
        
        # Simple binary classification problem
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf.fit(X, y)
        predictions = clf.predict(X)
        
        # Check predictions are binary
        assert set(predictions).issubset({0, 1})
        
        # Check we get reasonable predictions
        assert predictions[0] == 0 or predictions[1] == 0
        assert predictions[2] == 1 or predictions[3] == 1
    
    def test_method_chaining(self):
        """Test that fit returns self for method chaining."""
        clf = SVMClassifier(kernel='linear')
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        
        result = clf.fit(X, y)
        assert result is clf
        
        # Should be able to chain predict
        predictions = clf.fit(X, y).predict(X)
        assert len(predictions) == len(y)
