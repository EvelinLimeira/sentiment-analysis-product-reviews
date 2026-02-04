"""
SVM classifier module for sentiment analysis.

This module provides a wrapper around scikit-learn's SVM classifiers
for sentiment classification tasks.
"""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from typing import Union


class SVMClassifier:
    """SVM classifier with configurable kernel.
    
    This class wraps scikit-learn's SVM implementations to provide
    a consistent interface for sentiment classification. It supports
    both linear kernels (for BoW/TF-IDF features) and RBF kernels
    (for embedding features).
    """
    
    def __init__(self, kernel: str = 'linear', C: float = 1.0, gamma: str = 'scale'):
        """Initialize SVM classifier with specified parameters.
        
        Args:
            kernel: Kernel type - 'linear' for BoW/TF-IDF, 'rbf' for embeddings
            C: Regularization parameter (default 1.0)
            gamma: Kernel coefficient for RBF kernel (default 'scale')
        
        Raises:
            ValueError: If kernel is not 'linear' or 'rbf'
        """
        if kernel not in ['linear', 'rbf']:
            raise ValueError(f"Kernel must be 'linear' or 'rbf', got '{kernel}'")
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
        # Use LinearSVC for linear kernel (more efficient)
        # Use SVC for RBF kernel
        if kernel == 'linear':
            self.model = LinearSVC(C=C, random_state=42, max_iter=1000)
        else:
            self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Train SVM on training set.
        
        Args:
            X: Training features of shape (n_samples, n_features)
               Can be dense numpy array or sparse scipy matrix
            y: Training labels of shape (n_samples,)
        
        Returns:
            self: Returns the instance for method chaining
        
        Raises:
            ValueError: If X or y are empty or have mismatched shapes
        """
        # Handle both dense and sparse matrices
        n_samples_X = X.shape[0] if hasattr(X, 'shape') else len(X)
        n_samples_y = len(y)
        
        if n_samples_X == 0 or n_samples_y == 0:
            raise ValueError("Training data cannot be empty")
        
        if n_samples_X != n_samples_y:
            raise ValueError(f"X and y must have same length, got {n_samples_X} and {n_samples_y}")
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data.
        
        Args:
            X: Features of shape (n_samples, n_features)
               Can be dense numpy array or sparse scipy matrix
        
        Returns:
            Predicted labels of shape (n_samples,)
        
        Raises:
            ValueError: If X is empty
            RuntimeError: If model has not been fitted yet
        """
        # Handle both dense and sparse matrices
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        
        if n_samples == 0:
            raise ValueError("Input data cannot be empty")
        
        if not hasattr(self.model, 'classes_'):
            raise RuntimeError("Model must be fitted before calling predict()")
        
        return self.model.predict(X)
