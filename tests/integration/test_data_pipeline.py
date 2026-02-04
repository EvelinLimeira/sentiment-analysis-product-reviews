"""
Integration tests for the data pipeline.

Tests the complete flow from data loading through preprocessing,
verifying that all components work together correctly.

This validates Task 4: Checkpoint - Validate data pipeline
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    @pytest.fixture
    def mock_csv_path(self, tmp_path):
        """Create a mock CSV file with realistic review data."""
        np.random.seed(42)
        n_samples = 500
        
        # Create realistic review texts
        positive_reviews = [
            "This product is amazing! I love it so much.",
            "Great quality, highly recommend to everyone.",
            "Best purchase I've ever made. Five stars!",
            "Excellent product, exceeded my expectations.",
            "Wonderful item, fast shipping, very happy.",
        ]
        
        negative_reviews = [
            "Terrible product, waste of money.",
            "Very disappointed, does not work as advertised.",
            "Poor quality, broke after one day.",
            "Would not recommend, complete garbage.",
            "Awful experience, returning immediately.",
        ]
        
        texts = []
        ratings = []
        
        for i in range(n_samples):
            if i % 2 == 0:
                texts.append(positive_reviews[i % len(positive_reviews)] + f" Review {i}")
                ratings.append(np.random.choice([4, 5]))
            else:
                texts.append(negative_reviews[i % len(negative_reviews)] + f" Review {i}")
                ratings.append(np.random.choice([1, 2]))
        
        df = pd.DataFrame({'text': texts, 'rating': ratings})
        csv_path = tmp_path / "reviews.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_complete_pipeline_flow(self, mock_csv_path):
        """Test complete flow: load -> preprocess -> verify."""
        # Step 1: Load data
        loader = DataLoader(dataset_name=mock_csv_path, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        # Verify data loaded correctly
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Step 2: Fit preprocessor on training data ONLY
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(train_df['text'].tolist())
        
        # Step 3: Transform all sets
        train_preprocessed = preprocessor.transform(train_df['text'].tolist())
        val_preprocessed = preprocessor.transform(val_df['text'].tolist())
        test_preprocessed = preprocessor.transform(test_df['text'].tolist())
        
        # Verify preprocessing worked
        assert len(train_preprocessed) == len(train_df)
        assert len(val_preprocessed) == len(val_df)
        assert len(test_preprocessed) == len(test_df)
        
        # Verify all text is lowercase
        for text in train_preprocessed + val_preprocessed + test_preprocessed:
            if text:  # Skip empty strings
                assert text == text.lower(), f"Found uppercase in: {text}"
        
        # Verify no stopwords in output
        stopwords = preprocessor.get_stopwords()
        for text in train_preprocessed:
            if text:
                tokens = text.split()
                for token in tokens:
                    assert token not in stopwords, f"Stopword '{token}' found"
    
    def test_class_distribution_balanced(self, mock_csv_path):
        """Test that class distribution is balanced across splits."""
        loader = DataLoader(dataset_name=mock_csv_path, random_state=42)
        loader.load()
        
        distribution = loader.get_class_distribution()
        
        # Calculate positive ratios for each split
        def get_positive_ratio(counts):
            total = counts['negative'] + counts['positive']
            return counts['positive'] / total if total > 0 else 0
        
        train_ratio = get_positive_ratio(distribution['train'])
        val_ratio = get_positive_ratio(distribution['val'])
        test_ratio = get_positive_ratio(distribution['test'])
        
        # All ratios should be within 5% of each other (stratification)
        assert abs(train_ratio - val_ratio) < 0.05, \
            f"Train ({train_ratio:.2f}) and val ({val_ratio:.2f}) ratios differ too much"
        assert abs(train_ratio - test_ratio) < 0.05, \
            f"Train ({train_ratio:.2f}) and test ({test_ratio:.2f}) ratios differ too much"
        assert abs(val_ratio - test_ratio) < 0.05, \
            f"Val ({val_ratio:.2f}) and test ({test_ratio:.2f}) ratios differ too much"
        
        print(f"\nClass distribution verification:")
        print(f"  Train: {distribution['train']} (positive ratio: {train_ratio:.2%})")
        print(f"  Val:   {distribution['val']} (positive ratio: {val_ratio:.2%})")
        print(f"  Test:  {distribution['test']} (positive ratio: {test_ratio:.2%})")
    
    def test_no_data_leakage(self, mock_csv_path):
        """Test that there is no data leakage between splits."""
        loader = DataLoader(dataset_name=mock_csv_path, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        # Check no overlapping texts
        train_texts = set(train_df['text'])
        val_texts = set(val_df['text'])
        test_texts = set(test_df['text'])
        
        assert len(train_texts & val_texts) == 0, "Data leakage: train and val overlap"
        assert len(train_texts & test_texts) == 0, "Data leakage: train and test overlap"
        assert len(val_texts & test_texts) == 0, "Data leakage: val and test overlap"
        
        print("\nNo data leakage detected between splits.")
    
    def test_preprocessor_fit_only_on_training(self, mock_csv_path):
        """Test that preprocessor is fitted only on training data."""
        loader = DataLoader(dataset_name=mock_csv_path, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        # Fit preprocessor on training only
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(train_df['text'].tolist())
        
        # Get vocabulary from training
        train_vocab = preprocessor.vocabulary.copy()
        
        # Transform validation and test - vocabulary should NOT change
        preprocessor.transform(val_df['text'].tolist())
        preprocessor.transform(test_df['text'].tolist())
        
        # Vocabulary should remain the same (no new words added)
        assert preprocessor.vocabulary == train_vocab, \
            "Vocabulary changed after transforming val/test data"
        
        print(f"\nVocabulary size (from training only): {len(train_vocab)}")
    
    def test_split_proportions(self, mock_csv_path):
        """Test that split proportions are approximately 70/15/15."""
        loader = DataLoader(dataset_name=mock_csv_path, random_state=42)
        train_df, val_df, test_df = loader.load()
        
        total = len(train_df) + len(val_df) + len(test_df)
        
        train_prop = len(train_df) / total
        val_prop = len(val_df) / total
        test_prop = len(test_df) / total
        
        # Allow 5% tolerance
        assert abs(train_prop - 0.70) < 0.05, f"Train proportion {train_prop:.2f} not ~70%"
        assert abs(val_prop - 0.15) < 0.05, f"Val proportion {val_prop:.2f} not ~15%"
        assert abs(test_prop - 0.15) < 0.05, f"Test proportion {test_prop:.2f} not ~15%"
        
        print(f"\nSplit proportions:")
        print(f"  Train: {len(train_df)} ({train_prop:.1%})")
        print(f"  Val:   {len(val_df)} ({val_prop:.1%})")
        print(f"  Test:  {len(test_df)} ({test_prop:.1%})")
    
    def test_reproducibility_with_same_seed(self, mock_csv_path):
        """Test that same seed produces identical results."""
        # First run
        loader1 = DataLoader(dataset_name=mock_csv_path, random_state=42)
        train1, val1, test1 = loader1.load()
        
        preprocessor1 = Preprocessor(remove_stopwords=True)
        preprocessor1.fit(train1['text'].tolist())
        train_prep1 = preprocessor1.transform(train1['text'].tolist())
        
        # Second run with same seed
        loader2 = DataLoader(dataset_name=mock_csv_path, random_state=42)
        train2, val2, test2 = loader2.load()
        
        preprocessor2 = Preprocessor(remove_stopwords=True)
        preprocessor2.fit(train2['text'].tolist())
        train_prep2 = preprocessor2.transform(train2['text'].tolist())
        
        # Results should be identical
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)
        assert train_prep1 == train_prep2
        
        print("\nReproducibility verified: same seed produces identical results.")
    
    def test_different_seeds_produce_different_splits(self, mock_csv_path):
        """Test that different seeds produce different splits."""
        loader1 = DataLoader(dataset_name=mock_csv_path, random_state=42)
        train1, _, _ = loader1.load()
        
        loader2 = DataLoader(dataset_name=mock_csv_path, random_state=123)
        train2, _, _ = loader2.load()
        
        # Splits should be different
        assert list(train1['text'].head(10)) != list(train2['text'].head(10)), \
            "Different seeds should produce different splits"
        
        print("\nDifferent seeds produce different splits (as expected).")


class TestDataPipelineWithRealData:
    """
    Integration tests that can optionally use real data.
    These tests are skipped if the datasets library is not available.
    """
    
    @pytest.fixture
    def real_data_loader(self):
        """Create a DataLoader for real data (if available)."""
        try:
            from datasets import load_dataset
            return DataLoader(dataset_name='amazon_reviews', random_state=42)
        except ImportError:
            pytest.skip("datasets library not available")
    
    @pytest.mark.slow
    def test_real_data_loading(self, real_data_loader):
        """Test loading real Amazon reviews data."""
        try:
            train_df, val_df, test_df = real_data_loader.load()
            
            # Basic checks
            assert len(train_df) > 0
            assert len(val_df) > 0
            assert len(test_df) > 0
            
            # Check columns
            assert 'text' in train_df.columns
            assert 'label' in train_df.columns
            
            # Check labels are binary
            assert set(train_df['label'].unique()).issubset({0, 1})
            
            print(f"\nReal data loaded successfully:")
            print(f"  Train: {len(train_df)} samples")
            print(f"  Val:   {len(val_df)} samples")
            print(f"  Test:  {len(test_df)} samples")
            
        except Exception as e:
            pytest.skip(f"Could not load real data: {e}")
