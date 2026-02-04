"""
Integration test for Task 7: Checkpoint - Validate SVM models

This test validates that both SVM+BoW and SVM+Embeddings models work correctly
by training them on sample data and comparing their performance.

Task 7: Checkpoint - Validate SVM models
- Train and evaluate SVM+BoW and SVM+Embeddings
- Compare preliminary metrics
- Ensure all tests pass
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.vectorizers import BoWVectorizer
from src.embedding_encoder import EmbeddingEncoder
from src.classifiers import SVMClassifier


class TestCheckpointTask7:
    """Integration test for Task 7 checkpoint."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample sentiment data for testing."""
        np.random.seed(42)
        
        # Create realistic positive and negative reviews with unique variations
        positive_templates = [
            "This product is excellent and works perfectly",
            "Amazing quality, highly recommend to everyone",
            "Best purchase ever, very satisfied with it",
            "Great product, exceeded all my expectations",
            "Wonderful item, fast shipping and great service",
            "Love this product, it's fantastic and reliable",
            "Outstanding quality, worth every penny spent",
            "Superb product, exactly what I was looking for",
            "Excellent value, very happy with this purchase",
            "Perfect product, works great and looks amazing",
        ]
        
        negative_templates = [
            "Terrible product, complete waste of money",
            "Very disappointed, does not work at all",
            "Poor quality, broke after one day of use",
            "Would not recommend, absolutely horrible experience",
            "Awful product, returning it immediately today",
            "Worst purchase ever, total garbage and useless",
            "Horrible quality, not worth the price at all",
            "Disappointing product, failed to meet expectations",
            "Bad product, stopped working after few days",
            "Terrible experience, customer service was awful",
        ]
        
        # Create unique reviews by adding review numbers
        positive_reviews = [f"{template} Review {i}" for i, template in enumerate(positive_templates * 10)]
        negative_reviews = [f"{template} Review {i}" for i, template in enumerate(negative_templates * 10)]
        
        # Create DataFrame
        texts = positive_reviews + negative_reviews
        ratings = [5] * len(positive_reviews) + [1] * len(negative_reviews)
        
        df = pd.DataFrame({'text': texts, 'rating': ratings})
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        csv_path = tmp_path / "sample_reviews.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock embedding encoder with realistic embeddings."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 100
            
            # Create embeddings for common sentiment words
            np.random.seed(42)
            
            # Positive words - similar embeddings
            positive_base = np.random.randn(100).astype(np.float32)
            positive_words = [
                'excellent', 'amazing', 'best', 'great', 'wonderful',
                'love', 'outstanding', 'superb', 'perfect', 'fantastic',
                'reliable', 'quality', 'satisfied', 'happy', 'recommend',
                'worth', 'exceeded', 'expectations', 'value', 'works'
            ]
            
            # Negative words - similar to each other but different from positive
            negative_base = -positive_base + np.random.randn(100).astype(np.float32) * 0.5
            negative_words = [
                'terrible', 'disappointed', 'poor', 'horrible', 'awful',
                'worst', 'garbage', 'disappointing', 'bad', 'useless',
                'broke', 'failed', 'waste', 'returning', 'not'
            ]
            
            # Neutral words
            neutral_words = [
                'product', 'purchase', 'item', 'shipping', 'service',
                'price', 'day', 'experience', 'customer', 'this',
                'it', 'the', 'and', 'with', 'to', 'is', 'was',
                'after', 'all', 'my', 'very', 'one', 'ever', 'at'
            ]
            
            embeddings = {}
            
            # Create positive embeddings
            for word in positive_words:
                embeddings[word] = positive_base + np.random.randn(100).astype(np.float32) * 0.1
            
            # Create negative embeddings
            for word in negative_words:
                embeddings[word] = negative_base + np.random.randn(100).astype(np.float32) * 0.1
            
            # Create neutral embeddings
            for word in neutral_words:
                embeddings[word] = np.random.randn(100).astype(np.float32) * 0.3
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            mock_model.__len__ = Mock(return_value=len(embeddings))
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_task7_svm_bow_training_and_evaluation(self, sample_data):
        """
        Test SVM+BoW model training and evaluation.
        
        This validates:
        - Data loading and preprocessing
        - TF-IDF vectorization
        - SVM training with linear kernel
        - Model evaluation
        """
        # Step 1: Load data
        loader = DataLoader(dataset_name=sample_data, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        print(f"\nData loaded:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        # Step 2: Preprocess texts
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(train_df['text'].tolist())
        
        train_texts = preprocessor.transform(train_df['text'].tolist())
        test_texts = preprocessor.transform(test_df['text'].tolist())
        
        # Step 3: Vectorize with TF-IDF
        vectorizer = BoWVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"\nTF-IDF vectorization:")
        print(f"  Vocabulary size: {vectorizer.get_vocabulary_size()}")
        print(f"  Train shape: {X_train.shape}")
        print(f"  Test shape: {X_test.shape}")
        
        # Step 4: Train SVM with linear kernel (Requirement 3.4)
        clf = SVMClassifier(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)
        
        # Step 5: Evaluate
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"\nSVM+BoW Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Assertions
        assert accuracy > 0.5, "Accuracy should be better than random"
        assert f1 > 0.5, "F1-score should be better than random"
        assert 0.0 <= accuracy <= 1.0, "Accuracy should be in [0, 1]"
        assert 0.0 <= f1 <= 1.0, "F1-score should be in [0, 1]"
        
        # Should achieve reasonable performance on this simple dataset
        assert accuracy > 0.7, "Should achieve >70% accuracy on simple sentiment data"
    
    def test_task7_svm_embeddings_training_and_evaluation(self, sample_data, mock_encoder):
        """
        Test SVM+Embeddings model training and evaluation.
        
        This validates:
        - Data loading and preprocessing
        - Embedding encoding
        - SVM training with RBF kernel
        - Model evaluation
        """
        # Step 1: Load data
        loader = DataLoader(dataset_name=sample_data, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        # Step 2: Preprocess texts
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(train_df['text'].tolist())
        
        train_texts = preprocessor.transform(train_df['text'].tolist())
        test_texts = preprocessor.transform(test_df['text'].tolist())
        
        # Step 3: Encode with embeddings
        X_train = mock_encoder.encode_batch(train_texts)
        X_test = mock_encoder.encode_batch(test_texts)
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"\nEmbedding encoding:")
        print(f"  Embedding dim: {mock_encoder.get_embedding_dim()}")
        print(f"  Train shape: {X_train.shape}")
        print(f"  Test shape: {X_test.shape}")
        
        # Step 4: Train SVM with RBF kernel (Requirement 4.4)
        clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y_train)
        
        # Step 5: Evaluate
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"\nSVM+Embeddings Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Assertions
        assert accuracy > 0.5, "Accuracy should be better than random"
        assert f1 > 0.5, "F1-score should be better than random"
        assert 0.0 <= accuracy <= 1.0, "Accuracy should be in [0, 1]"
        assert 0.0 <= f1 <= 1.0, "F1-score should be in [0, 1]"
        
        # Should achieve reasonable performance
        assert accuracy > 0.6, "Should achieve >60% accuracy with embeddings"
    
    def test_task7_compare_svm_bow_vs_embeddings(self, sample_data, mock_encoder):
        """
        Compare SVM+BoW vs SVM+Embeddings performance.
        
        This is the main checkpoint test that validates both models
        work correctly and can be compared.
        """
        # Load and prepare data
        loader = DataLoader(dataset_name=sample_data, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(train_df['text'].tolist())
        
        train_texts = preprocessor.transform(train_df['text'].tolist())
        test_texts = preprocessor.transform(test_df['text'].tolist())
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        # Train SVM+BoW
        vectorizer = BoWVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_bow = vectorizer.fit_transform(train_texts)
        X_test_bow = vectorizer.transform(test_texts)
        
        clf_bow = SVMClassifier(kernel='linear', C=1.0)
        clf_bow.fit(X_train_bow, y_train)
        y_pred_bow = clf_bow.predict(X_test_bow)
        
        # Train SVM+Embeddings
        X_train_emb = mock_encoder.encode_batch(train_texts)
        X_test_emb = mock_encoder.encode_batch(test_texts)
        
        clf_emb = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf_emb.fit(X_train_emb, y_train)
        y_pred_emb = clf_emb.predict(X_test_emb)
        
        # Calculate metrics for both
        results = {
            'SVM+BoW': {
                'accuracy': accuracy_score(y_test, y_pred_bow),
                'f1': f1_score(y_test, y_pred_bow, average='weighted'),
                'precision': precision_score(y_test, y_pred_bow, average='weighted'),
                'recall': recall_score(y_test, y_pred_bow, average='weighted'),
            },
            'SVM+Embeddings': {
                'accuracy': accuracy_score(y_test, y_pred_emb),
                'f1': f1_score(y_test, y_pred_emb, average='weighted'),
                'precision': precision_score(y_test, y_pred_emb, average='weighted'),
                'recall': recall_score(y_test, y_pred_emb, average='weighted'),
            }
        }
        
        # Print comparison
        print("\n" + "="*60)
        print("TASK 7 CHECKPOINT: SVM Models Comparison")
        print("="*60)
        print(f"\n{'Metric':<15} {'SVM+BoW':<15} {'SVM+Embeddings':<15}")
        print("-"*60)
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            bow_val = results['SVM+BoW'][metric]
            emb_val = results['SVM+Embeddings'][metric]
            print(f"{metric.capitalize():<15} {bow_val:<15.4f} {emb_val:<15.4f}")
        print("="*60)
        
        # Validate both models work
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                assert 0.0 <= value <= 1.0, \
                    f"{model_name} {metric_name} should be in [0, 1], got {value}"
                assert value > 0.5, \
                    f"{model_name} {metric_name} should be better than random, got {value}"
        
        # Both models should achieve reasonable performance
        assert results['SVM+BoW']['accuracy'] > 0.7, \
            "SVM+BoW should achieve >70% accuracy"
        assert results['SVM+Embeddings']['accuracy'] > 0.6, \
            "SVM+Embeddings should achieve >60% accuracy"
        
        print("\n✓ Task 7 Checkpoint PASSED: Both SVM models work correctly!")
    
    def test_task7_models_use_correct_kernels(self, sample_data, mock_encoder):
        """Verify that correct kernels are used for each model."""
        # Load minimal data
        loader = DataLoader(dataset_name=sample_data, random_state=42)
        train_df, _, _ = loader.load()
        
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(train_df['text'].tolist())
        train_texts = preprocessor.transform(train_df['text'].tolist())
        y_train = train_df['label'].values
        
        # SVM+BoW should use linear kernel
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        X_train_bow = vectorizer.fit_transform(train_texts)
        clf_bow = SVMClassifier(kernel='linear', C=1.0)
        clf_bow.fit(X_train_bow, y_train)
        
        assert clf_bow.kernel == 'linear', "SVM+BoW should use linear kernel"
        
        # SVM+Embeddings should use RBF kernel
        X_train_emb = mock_encoder.encode_batch(train_texts)
        clf_emb = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        clf_emb.fit(X_train_emb, y_train)
        
        assert clf_emb.kernel == 'rbf', "SVM+Embeddings should use RBF kernel"
        assert clf_emb.gamma == 'scale', "SVM+Embeddings should use gamma='scale'"
        
        print("\n✓ Correct kernels verified:")
        print("  - SVM+BoW: linear kernel")
        print("  - SVM+Embeddings: RBF kernel with gamma='scale'")
    
    def test_task7_no_data_leakage_in_pipeline(self, sample_data, mock_encoder):
        """Verify no data leakage in the complete pipeline."""
        # Load data
        loader = DataLoader(dataset_name=sample_data, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        # Verify no overlap between splits
        train_texts = set(train_df['text'])
        val_texts = set(val_df['text'])
        test_texts = set(test_df['text'])
        
        assert len(train_texts & val_texts) == 0, "Train and val should not overlap"
        assert len(train_texts & test_texts) == 0, "Train and test should not overlap"
        assert len(val_texts & test_texts) == 0, "Val and test should not overlap"
        
        # Verify preprocessor is fitted only on training
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(train_df['text'].tolist())
        train_vocab_size = preprocessor.get_vocabulary_size()
        
        # Transform test data - vocabulary should not change
        preprocessor.transform(test_df['text'].tolist())
        assert preprocessor.get_vocabulary_size() == train_vocab_size, \
            "Vocabulary should not change after transforming test data"
        
        # Verify vectorizer is fitted only on training
        train_texts_prep = preprocessor.transform(train_df['text'].tolist())
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(train_texts_prep)
        vocab_size = vectorizer.get_vocabulary_size()
        
        # Transform test data - vocabulary should not change
        test_texts_prep = preprocessor.transform(test_df['text'].tolist())
        vectorizer.transform(test_texts_prep)
        assert vectorizer.get_vocabulary_size() == vocab_size, \
            "TF-IDF vocabulary should not change after transforming test data"
        
        print("\n✓ No data leakage detected in pipeline:")
        print("  - No overlap between train/val/test splits")
        print("  - Preprocessor fitted only on training data")
        print("  - Vectorizer fitted only on training data")


class TestTask7MetricsValidation:
    """Validate that metrics are calculated correctly."""
    
    def test_metrics_in_valid_range(self):
        """Test that all metrics are in valid range [0, 1]."""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        assert accuracy == 1.0
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0
        
        # Worst predictions
        y_pred_worst = np.array([1, 1, 0, 0])
        
        accuracy_worst = accuracy_score(y_true, y_pred_worst)
        f1_worst = f1_score(y_true, y_pred_worst, average='weighted')
        
        assert accuracy_worst == 0.0
        assert 0.0 <= f1_worst <= 1.0
        
        print("\n✓ Metrics validation passed:")
        print(f"  - Perfect predictions: accuracy={accuracy}, f1={f1}")
        print(f"  - Worst predictions: accuracy={accuracy_worst}, f1={f1_worst:.4f}")
    
    def test_confusion_matrix_validity(self):
        """Test confusion matrix properties."""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Sum should equal total samples
        assert cm.sum() == len(y_true)
        
        # All elements should be non-negative
        assert np.all(cm >= 0)
        
        # Should be 2x2 for binary classification
        assert cm.shape == (2, 2)
        
        print(f"\n✓ Confusion matrix validation passed:")
        print(f"  Shape: {cm.shape}")
        print(f"  Sum: {cm.sum()} (equals {len(y_true)} samples)")
        print(f"  Matrix:\n{cm}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
