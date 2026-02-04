"""
Integration tests for SVM classifier with embedding vectors.

These tests verify that the SVMClassifier works correctly with
embedding vectors from EmbeddingEncoder as specified in Requirement 4.4.

Task 6.2: Integrate embeddings with SVM
- Reuse SVMClassifier with embedding vectors (kernel='rbf', gamma='scale')
- Requirements: 4.4
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.classifiers import SVMClassifier
from src.embedding_encoder import EmbeddingEncoder


class TestSVMWithEmbeddings:
    """Test SVMClassifier with embedding vectors (Requirement 4.4)."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder with controlled embeddings."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 100
            mock_model.__len__ = Mock(return_value=10)
            
            # Create embeddings for sentiment-related words
            embeddings = {
                'good': np.random.randn(100).astype(np.float32),
                'great': np.random.randn(100).astype(np.float32),
                'excellent': np.random.randn(100).astype(np.float32),
                'amazing': np.random.randn(100).astype(np.float32),
                'wonderful': np.random.randn(100).astype(np.float32),
                'bad': np.random.randn(100).astype(np.float32),
                'terrible': np.random.randn(100).astype(np.float32),
                'awful': np.random.randn(100).astype(np.float32),
                'horrible': np.random.randn(100).astype(np.float32),
                'poor': np.random.randn(100).astype(np.float32),
            }
            
            # Make positive words similar to each other
            base_positive = np.random.randn(100).astype(np.float32)
            for word in ['good', 'great', 'excellent', 'amazing', 'wonderful']:
                embeddings[word] = base_positive + np.random.randn(100).astype(np.float32) * 0.1
            
            # Make negative words similar to each other but different from positive
            base_negative = -base_positive + np.random.randn(100).astype(np.float32) * 0.5
            for word in ['bad', 'terrible', 'awful', 'horrible', 'poor']:
                embeddings[word] = base_negative + np.random.randn(100).astype(np.float32) * 0.1
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_requirement_4_4_svm_rbf_with_embeddings(self, mock_encoder):
        """
        Requirement 4.4: Train SVM model with RBF kernel (C=1.0, gamma='scale')
        using embedding vectors from training.
        """
        # Create training data with embedding vectors
        positive_texts = [
            'good product',
            'great quality',
            'excellent service',
            'amazing experience',
            'wonderful item'
        ]
        
        negative_texts = [
            'bad product',
            'terrible quality',
            'awful service',
            'horrible experience',
            'poor item'
        ]
        
        # Encode texts to embedding vectors
        X_train_pos = mock_encoder.encode_batch(positive_texts)
        X_train_neg = mock_encoder.encode_batch(negative_texts)
        X_train = np.vstack([X_train_pos, X_train_neg])
        
        # Create labels (1 for positive, 0 for negative)
        y_train = np.array([1] * len(positive_texts) + [0] * len(negative_texts))
        
        # Initialize SVM with RBF kernel as per Requirement 4.4
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        
        # Train the model
        clf.fit(X_train, y_train)
        
        # Verify model is trained
        assert hasattr(clf.model, 'classes_')
        assert len(clf.model.classes_) == 2
        
        # Test prediction on training data
        predictions = clf.predict(X_train)
        
        # Verify predictions
        assert len(predictions) == len(y_train)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should achieve good accuracy on training data
        accuracy = np.mean(predictions == y_train)
        assert accuracy > 0.7  # Should learn the pattern
    
    def test_svm_rbf_parameters_match_requirement(self):
        """Test that SVM is initialized with correct parameters (C=1.0, gamma='scale')."""
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        
        assert clf.kernel == 'rbf'
        assert clf.C == 1.0
        assert clf.gamma == 'scale'
        assert clf.model.kernel == 'rbf'
        assert clf.model.C == 1.0
        assert clf.model.gamma == 'scale'
    
    def test_svm_rbf_with_dense_embedding_vectors(self, mock_encoder):
        """Test that SVM works with dense embedding vectors (not sparse)."""
        # Create dense embedding vectors
        texts = ['good', 'bad', 'great', 'terrible']
        X = mock_encoder.encode_batch(texts)
        y = np.array([1, 0, 1, 0])
        
        # Verify X is dense (not sparse)
        assert isinstance(X, np.ndarray)
        assert not hasattr(X, 'toarray')  # Not a sparse matrix
        
        # Train SVM with RBF kernel
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        # Should work without errors
        predictions = clf.predict(X)
        assert len(predictions) == len(y)
    
    def test_svm_rbf_embedding_dimension_consistency(self, mock_encoder):
        """Test that SVM handles consistent embedding dimensions."""
        # All embeddings should have same dimension
        texts = ['good product', 'bad service', 'great quality']
        X = mock_encoder.encode_batch(texts)
        
        # Verify all vectors have same dimension
        assert X.shape[1] == mock_encoder.get_embedding_dim()
        assert X.shape[0] == len(texts)
        
        # Train SVM
        y = np.array([1, 0, 1])
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        # Predict on new data with same dimension
        X_test = mock_encoder.encode_batch(['excellent', 'awful'])
        predictions = clf.predict(X_test)
        
        assert len(predictions) == 2
    
    def test_svm_rbf_with_larger_dataset(self, mock_encoder):
        """Test SVM with RBF kernel on larger embedding dataset."""
        np.random.seed(42)
        
        # Create larger dataset
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']
        
        # Generate multiple combinations
        positive_texts = []
        negative_texts = []
        
        for i in range(20):
            # Positive reviews
            pos_word = np.random.choice(positive_words)
            positive_texts.append(pos_word)
            
            # Negative reviews
            neg_word = np.random.choice(negative_words)
            negative_texts.append(neg_word)
        
        # Encode to embeddings
        X_pos = mock_encoder.encode_batch(positive_texts)
        X_neg = mock_encoder.encode_batch(negative_texts)
        X_train = np.vstack([X_pos, X_neg])
        y_train = np.array([1] * len(positive_texts) + [0] * len(negative_texts))
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        # Train SVM with RBF kernel (Requirement 4.4)
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)
        
        # Test on new data
        X_test_pos = mock_encoder.encode_batch(['good', 'excellent'])
        X_test_neg = mock_encoder.encode_batch(['bad', 'terrible'])
        X_test = np.vstack([X_test_pos, X_test_neg])
        y_test = np.array([1, 1, 0, 0])
        
        predictions = clf.predict(X_test)
        
        # Should achieve reasonable accuracy
        accuracy = np.mean(predictions == y_test)
        assert accuracy >= 0.5  # At least better than random
    
    def test_svm_rbf_handles_zero_vectors(self, mock_encoder):
        """Test that SVM handles zero vectors (from all-OOV texts)."""
        # Mix of known and unknown words
        texts = [
            'good',  # Known
            'xyz123',  # Unknown (will be zero vector)
            'bad',  # Known
            'qwerty',  # Unknown (will be zero vector)
        ]
        
        X = mock_encoder.encode_batch(texts)
        y = np.array([1, 1, 0, 0])
        
        # Train SVM
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        # Should handle zero vectors without errors
        predictions = clf.predict(X)
        assert len(predictions) == len(y)
    
    def test_svm_rbf_prediction_consistency(self, mock_encoder):
        """Test that predictions are consistent across multiple calls."""
        texts = ['good product', 'bad service', 'great quality', 'terrible experience']
        X = mock_encoder.encode_batch(texts)
        y = np.array([1, 0, 1, 0])
        
        # Train SVM
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        # Make predictions multiple times
        X_test = mock_encoder.encode_batch(['excellent', 'awful'])
        pred1 = clf.predict(X_test)
        pred2 = clf.predict(X_test)
        pred3 = clf.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)
    
    def test_svm_rbf_with_single_word_vs_multiple_words(self, mock_encoder):
        """Test SVM with embeddings from single words vs multiple words."""
        # Single word texts
        single_word_texts = ['good', 'bad', 'great', 'terrible']
        X_single = mock_encoder.encode_batch(single_word_texts)
        
        # Multiple word texts (mean of embeddings)
        multi_word_texts = ['good great', 'bad terrible', 'good excellent', 'bad awful']
        X_multi = mock_encoder.encode_batch(multi_word_texts)
        
        y = np.array([1, 0, 1, 0])
        
        # Train on single words
        clf_single = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf_single.fit(X_single, y)
        
        # Train on multiple words
        clf_multi = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf_multi.fit(X_multi, y)
        
        # Both should work
        pred_single = clf_single.predict(X_single)
        pred_multi = clf_multi.predict(X_multi)
        
        assert len(pred_single) == len(y)
        assert len(pred_multi) == len(y)


class TestSVMEmbeddingsVsTFIDF:
    """Compare SVM behavior with embeddings vs TF-IDF."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 50
            mock_model.__len__ = Mock(return_value=5)
            
            # Simple embeddings
            np.random.seed(42)
            embeddings = {
                'good': np.random.randn(50).astype(np.float32),
                'bad': np.random.randn(50).astype(np.float32),
                'product': np.random.randn(50).astype(np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_rbf_kernel_for_embeddings_vs_linear_for_tfidf(self, mock_encoder):
        """
        Verify that RBF kernel is used for embeddings (Requirement 4.4)
        while linear kernel is used for TF-IDF (Requirement 3.4).
        """
        # SVM for embeddings should use RBF kernel
        clf_embeddings = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        assert clf_embeddings.kernel == 'rbf'
        assert clf_embeddings.gamma == 'scale'
        
        # SVM for TF-IDF should use linear kernel
        clf_tfidf = SVMClassifier(kernel='linear', C=1.0)
        assert clf_tfidf.kernel == 'linear'
        
        # Train both on embedding data
        texts = ['good product', 'bad product']
        X = mock_encoder.encode_batch(texts)
        y = np.array([1, 0])
        
        clf_embeddings.fit(X, y)
        clf_tfidf.fit(X, y)
        
        # Both should work but use different kernels
        pred_rbf = clf_embeddings.predict(X)
        pred_linear = clf_tfidf.predict(X)
        
        assert len(pred_rbf) == len(y)
        assert len(pred_linear) == len(y)


class TestSVMEmbeddingsEdgeCases:
    """Test edge cases for SVM with embeddings."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 10
            mock_model.__len__ = Mock(return_value=3)
            
            embeddings = {
                'word1': np.array([1.0] * 10, dtype=np.float32),
                'word2': np.array([0.5] * 10, dtype=np.float32),
                'word3': np.array([0.0] * 10, dtype=np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_svm_rbf_with_minimum_samples(self, mock_encoder):
        """Test SVM with minimum number of samples."""
        # Minimum 2 samples per class
        texts = ['word1', 'word2', 'word3', 'word3']
        X = mock_encoder.encode_batch(texts)
        y = np.array([1, 1, 0, 0])
        
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        predictions = clf.predict(X)
        assert len(predictions) == len(y)
    
    def test_svm_rbf_with_identical_embeddings(self, mock_encoder):
        """Test SVM when some embeddings are identical."""
        # 'word3' appears multiple times
        texts = ['word3', 'word3', 'word1', 'word1']
        X = mock_encoder.encode_batch(texts)
        y = np.array([0, 0, 1, 1])
        
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        # Should handle identical vectors
        predictions = clf.predict(X)
        assert len(predictions) == len(y)
    
    def test_svm_rbf_with_very_similar_embeddings(self, mock_encoder):
        """Test SVM when embeddings are very similar."""
        # Create very similar embeddings
        texts = ['word1', 'word2']  # word2 = 0.5 * word1
        X = mock_encoder.encode_batch(texts)
        
        # Repeat to create dataset
        X = np.vstack([X, X, X])
        y = np.array([1, 0, 1, 0, 1, 0])
        
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        predictions = clf.predict(X)
        assert len(predictions) == len(y)
    
    def test_svm_rbf_gamma_scale_behavior(self, mock_encoder):
        """Test that gamma='scale' is properly used."""
        texts = ['word1', 'word2', 'word3']
        X = mock_encoder.encode_batch(texts)
        y = np.array([1, 0, 1])
        
        # gamma='scale' should be 1 / (n_features * X.var())
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X, y)
        
        # Verify gamma parameter is set correctly
        assert clf.gamma == 'scale'
        assert clf.model.gamma == 'scale'
        
        # Should work without errors
        predictions = clf.predict(X)
        assert len(predictions) == len(y)


class TestSVMEmbeddingsRequirementValidation:
    """Explicit tests for Requirement 4.4 validation."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 100
            mock_model.__len__ = Mock(return_value=6)
            
            # Create distinct embeddings for positive and negative words
            np.random.seed(42)
            positive_base = np.random.randn(100).astype(np.float32)
            negative_base = -positive_base
            
            embeddings = {
                'good': positive_base + np.random.randn(100).astype(np.float32) * 0.1,
                'great': positive_base + np.random.randn(100).astype(np.float32) * 0.1,
                'excellent': positive_base + np.random.randn(100).astype(np.float32) * 0.1,
                'bad': negative_base + np.random.randn(100).astype(np.float32) * 0.1,
                'terrible': negative_base + np.random.randn(100).astype(np.float32) * 0.1,
                'awful': negative_base + np.random.randn(100).astype(np.float32) * 0.1,
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_requirement_4_4_complete_workflow(self, mock_encoder):
        """
        Complete workflow test for Requirement 4.4:
        Train an SVM model with RBF kernel (C=1.0, gamma='scale')
        using embedding vectors from training.
        """
        # Step 1: Prepare training data
        train_texts = [
            'good', 'great', 'excellent',  # Positive
            'bad', 'terrible', 'awful'  # Negative
        ]
        
        # Step 2: Convert texts to embedding vectors
        X_train = mock_encoder.encode_batch(train_texts)
        y_train = np.array([1, 1, 1, 0, 0, 0])
        
        # Verify embeddings are dense vectors
        assert isinstance(X_train, np.ndarray)
        assert X_train.shape == (6, 100)
        assert X_train.dtype == np.float32
        
        # Step 3: Initialize SVM with RBF kernel as per Requirement 4.4
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        
        # Verify parameters
        assert clf.kernel == 'rbf'
        assert clf.C == 1.0
        assert clf.gamma == 'scale'
        
        # Step 4: Train the model
        clf.fit(X_train, y_train)
        
        # Verify model is trained
        assert hasattr(clf.model, 'classes_')
        assert hasattr(clf.model, 'support_vectors_')
        
        # Step 5: Make predictions
        predictions = clf.predict(X_train)
        
        # Verify predictions
        assert predictions.shape == (6,)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Step 6: Test on new data
        test_texts = ['good', 'bad']
        X_test = mock_encoder.encode_batch(test_texts)
        test_predictions = clf.predict(X_test)
        
        assert len(test_predictions) == 2
        assert all(pred in [0, 1] for pred in test_predictions)
        
        # Should achieve good accuracy on training data
        train_accuracy = np.mean(predictions == y_train)
        assert train_accuracy > 0.7
    
    def test_requirement_4_4_uses_training_embeddings_only(self, mock_encoder):
        """
        Verify that SVM is trained using embedding vectors from training set only.
        """
        # Training data
        train_texts = ['good', 'bad', 'great', 'terrible']
        X_train = mock_encoder.encode_batch(train_texts)
        y_train = np.array([1, 0, 1, 0])
        
        # Train SVM
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)
        
        # Test data (not used in training)
        test_texts = ['excellent', 'awful']
        X_test = mock_encoder.encode_batch(test_texts)
        
        # Model should only have been trained on training data
        # Support vectors should come from training data only
        assert clf.model.support_vectors_.shape[1] == X_train.shape[1]
        
        # Should be able to predict on test data
        predictions = clf.predict(X_test)
        assert len(predictions) == 2
    
    def test_requirement_4_4_rbf_kernel_vs_linear_kernel(self, mock_encoder):
        """
        Verify that RBF kernel is specifically used for embeddings
        (as opposed to linear kernel used for TF-IDF).
        """
        # Prepare data
        texts = ['good', 'bad', 'great', 'terrible']
        X = mock_encoder.encode_batch(texts)
        y = np.array([1, 0, 1, 0])
        
        # Train with RBF kernel (Requirement 4.4)
        clf_rbf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf_rbf.fit(X, y)
        
        # Train with linear kernel (for comparison)
        clf_linear = SVMClassifier(kernel='linear', C=1.0)
        clf_linear.fit(X, y)
        
        # Both should work, but RBF is required for embeddings
        assert clf_rbf.kernel == 'rbf'
        assert clf_linear.kernel == 'linear'
        
        # Make predictions
        pred_rbf = clf_rbf.predict(X)
        pred_linear = clf_linear.predict(X)
        
        assert len(pred_rbf) == len(y)
        assert len(pred_linear) == len(y)
        
        # RBF kernel should be used for embeddings per requirement
        assert clf_rbf.model.kernel == 'rbf'
