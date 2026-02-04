"""
Unit tests for the DataLoader class.

Tests cover:
- Label conversion (ratings to binary labels)
- Stratified train/validation/test splits
- Class distribution reporting
- Multiple random seeds support
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    DataLoader,
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidDataFormatError
)


class TestConvertRatingToLabel:
    """Tests for convert_rating_to_label method."""
    
    def test_rating_1_returns_negative(self):
        """Rating 1 should return 0 (negative)."""
        loader = DataLoader()
        assert loader.convert_rating_to_label(1) == 0
    
    def test_rating_2_returns_negative(self):
        """Rating 2 should return 0 (negative)."""
        loader = DataLoader()
        assert loader.convert_rating_to_label(2) == 0
    
    def test_rating_3_returns_none(self):
        """Rating 3 should return None (neutral, to be discarded)."""
        loader = DataLoader()
        assert loader.convert_rating_to_label(3) is None
    
    def test_rating_4_returns_positive(self):
        """Rating 4 should return 1 (positive)."""
        loader = DataLoader()
        assert loader.convert_rating_to_label(4) == 1
    
    def test_rating_5_returns_positive(self):
        """Rating 5 should return 1 (positive)."""
        loader = DataLoader()
        assert loader.convert_rating_to_label(5) == 1
    
    def test_invalid_rating_returns_none(self):
        """Invalid ratings should return None."""
        loader = DataLoader()
        assert loader.convert_rating_to_label(0) is None
        assert loader.convert_rating_to_label(6) is None
        assert loader.convert_rating_to_label(-1) is None


class TestDataLoaderInit:
    """Tests for DataLoader initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        loader = DataLoader()
        assert loader.dataset_name == 'amazon_reviews'
        assert loader.test_size == 0.15
        assert loader.val_size == 0.15
        assert loader.random_state == 42
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        loader = DataLoader(
            dataset_name='custom_dataset',
            test_size=0.2,
            val_size=0.1,
            random_state=123
        )
        assert loader.dataset_name == 'custom_dataset'
        assert loader.test_size == 0.2
        assert loader.val_size == 0.1
        assert loader.random_state == 123
    
    def test_invalid_split_sizes_raises_error(self):
        """Invalid split sizes should raise ValueError."""
        with pytest.raises(ValueError):
            DataLoader(test_size=0.5, val_size=0.6)  # Sum > 1
        
        with pytest.raises(ValueError):
            DataLoader(test_size=0, val_size=0.15)  # Zero test size
        
        with pytest.raises(ValueError):
            DataLoader(test_size=0.15, val_size=0)  # Zero val size


class TestDataLoaderWithMockData:
    """Tests using mock data to avoid network dependencies."""
    
    @pytest.fixture
    def mock_reviews_df(self):
        """Create a mock reviews DataFrame."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create balanced dataset with ratings 1-5
        ratings = np.random.choice([1, 2, 4, 5], size=n_samples)
        texts = [f"This is review number {i}" for i in range(n_samples)]
        
        return pd.DataFrame({
            'text': texts,
            'rating': ratings
        })
    
    @pytest.fixture
    def loader_with_mock_data(self, mock_reviews_df, tmp_path):
        """Create a DataLoader with mock CSV data."""
        csv_path = tmp_path / "mock_reviews.csv"
        mock_reviews_df.to_csv(csv_path, index=False)
        return DataLoader(dataset_name=str(csv_path))
    
    def test_load_returns_three_dataframes(self, loader_with_mock_data):
        """load() should return train, val, test DataFrames."""
        train_df, val_df, test_df = loader_with_mock_data.load()
        
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
    
    def test_dataframes_have_correct_columns(self, loader_with_mock_data):
        """DataFrames should have 'text' and 'label' columns."""
        train_df, val_df, test_df = loader_with_mock_data.load()
        
        expected_columns = {'text', 'label'}
        assert set(train_df.columns) == expected_columns
        assert set(val_df.columns) == expected_columns
        assert set(test_df.columns) == expected_columns
    
    def test_labels_are_binary(self, loader_with_mock_data):
        """Labels should only be 0 or 1."""
        train_df, val_df, test_df = loader_with_mock_data.load()
        
        all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']])
        assert set(all_labels.unique()) == {0, 1}
    
    def test_split_proportions_approximately_correct(self, loader_with_mock_data):
        """Split proportions should be approximately 70/15/15."""
        train_df, val_df, test_df = loader_with_mock_data.load()
        
        total = len(train_df) + len(val_df) + len(test_df)
        
        train_prop = len(train_df) / total
        val_prop = len(val_df) / total
        test_prop = len(test_df) / total
        
        # Allow 5% tolerance
        assert abs(train_prop - 0.70) < 0.05
        assert abs(val_prop - 0.15) < 0.05
        assert abs(test_prop - 0.15) < 0.05
    
    def test_stratification_preserves_class_distribution(self, loader_with_mock_data):
        """Stratified split should preserve class distribution across splits."""
        train_df, val_df, test_df = loader_with_mock_data.load()
        
        def get_positive_ratio(df):
            return df['label'].mean()
        
        train_ratio = get_positive_ratio(train_df)
        val_ratio = get_positive_ratio(val_df)
        test_ratio = get_positive_ratio(test_df)
        
        # All ratios should be within 5% of each other
        assert abs(train_ratio - val_ratio) < 0.05
        assert abs(train_ratio - test_ratio) < 0.05
        assert abs(val_ratio - test_ratio) < 0.05
    
    def test_no_data_leakage_between_splits(self, loader_with_mock_data):
        """There should be no overlapping samples between splits."""
        train_df, val_df, test_df = loader_with_mock_data.load()
        
        train_texts = set(train_df['text'])
        val_texts = set(val_df['text'])
        test_texts = set(test_df['text'])
        
        # No overlap between any splits
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0


class TestClassDistribution:
    """Tests for get_class_distribution method."""
    
    @pytest.fixture
    def loaded_loader(self, tmp_path):
        """Create a loaded DataLoader with mock data."""
        np.random.seed(42)
        n_samples = 500
        
        ratings = np.random.choice([1, 2, 4, 5], size=n_samples)
        texts = [f"Review {i}" for i in range(n_samples)]
        
        df = pd.DataFrame({'text': texts, 'rating': ratings})
        csv_path = tmp_path / "reviews.csv"
        df.to_csv(csv_path, index=False)
        
        loader = DataLoader(dataset_name=str(csv_path))
        loader.load()
        return loader
    
    def test_get_class_distribution_structure(self, loaded_loader):
        """get_class_distribution should return correct structure."""
        distribution = loaded_loader.get_class_distribution()
        
        assert 'train' in distribution
        assert 'val' in distribution
        assert 'test' in distribution
        
        for split in ['train', 'val', 'test']:
            assert 'negative' in distribution[split]
            assert 'positive' in distribution[split]
    
    def test_get_class_distribution_counts_are_integers(self, loaded_loader):
        """Class counts should be integers."""
        distribution = loaded_loader.get_class_distribution()
        
        for split in ['train', 'val', 'test']:
            assert isinstance(distribution[split]['negative'], int)
            assert isinstance(distribution[split]['positive'], int)
    
    def test_get_class_distribution_before_load_raises_error(self):
        """Calling get_class_distribution before load should raise error."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="Data has not been loaded"):
            loader.get_class_distribution()


class TestMultipleRandomSeeds:
    """Tests for multiple random seeds support."""
    
    @pytest.fixture
    def csv_path(self, tmp_path):
        """Create a mock CSV file."""
        np.random.seed(42)
        n_samples = 500
        
        ratings = np.random.choice([1, 2, 4, 5], size=n_samples)
        texts = [f"Review {i}" for i in range(n_samples)]
        
        df = pd.DataFrame({'text': texts, 'rating': ratings})
        csv_path = tmp_path / "reviews.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_different_seeds_produce_different_splits(self, csv_path):
        """Different random seeds should produce different splits."""
        loader1 = DataLoader(dataset_name=csv_path, random_state=42)
        train1, _, _ = loader1.load()
        
        loader2 = DataLoader(dataset_name=csv_path, random_state=123)
        train2, _, _ = loader2.load()
        
        # The splits should be different
        # Compare first few texts
        assert list(train1['text'].head(10)) != list(train2['text'].head(10))
    
    def test_same_seed_produces_same_splits(self, csv_path):
        """Same random seed should produce identical splits."""
        loader1 = DataLoader(dataset_name=csv_path, random_state=42)
        train1, val1, test1 = loader1.load()
        
        loader2 = DataLoader(dataset_name=csv_path, random_state=42)
        train2, val2, test2 = loader2.load()
        
        # The splits should be identical
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)
    
    def test_set_random_state_clears_cached_data(self, csv_path):
        """set_random_state should clear cached data."""
        loader = DataLoader(dataset_name=csv_path, random_state=42)
        loader.load()
        
        assert loader.train_df is not None
        
        loader.set_random_state(123)
        
        assert loader.train_df is None
        assert loader.val_df is None
        assert loader.test_df is None


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_csv_raises_error(self, tmp_path):
        """Empty CSV should raise EmptyDatasetError."""
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame({'text': [], 'rating': []}).to_csv(csv_path, index=False)
        
        loader = DataLoader(dataset_name=str(csv_path))
        
        with pytest.raises(EmptyDatasetError):
            loader.load()
    
    def test_all_neutral_ratings_raises_error(self, tmp_path):
        """Dataset with only rating 3 should raise EmptyDatasetError."""
        csv_path = tmp_path / "neutral.csv"
        df = pd.DataFrame({
            'text': ['Review 1', 'Review 2', 'Review 3'],
            'rating': [3, 3, 3]
        })
        df.to_csv(csv_path, index=False)
        
        loader = DataLoader(dataset_name=str(csv_path))
        
        with pytest.raises(EmptyDatasetError):
            loader.load()
    
    def test_missing_columns_raises_error(self, tmp_path):
        """CSV with missing columns should raise InvalidDataFormatError."""
        csv_path = tmp_path / "missing_cols.csv"
        df = pd.DataFrame({
            'wrong_column': ['text1', 'text2'],
            'another_wrong': [1, 2]
        })
        df.to_csv(csv_path, index=False)
        
        loader = DataLoader(dataset_name=str(csv_path))
        
        with pytest.raises(InvalidDataFormatError):
            loader.load()
    
    def test_nonexistent_csv_raises_error(self):
        """Non-existent CSV file should raise DatasetNotFoundError."""
        loader = DataLoader(dataset_name='/nonexistent/path/file.csv')
        
        with pytest.raises(DatasetNotFoundError):
            loader.load()
