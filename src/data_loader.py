"""
Data loading module for sentiment analysis NLP project.

This module provides the DataLoader class for loading and preparing product reviews
datasets with proper train/validation/test splits and label conversion.

"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when the requested dataset cannot be found or loaded."""
    pass


class EmptyDatasetError(Exception):
    """Raised when the dataset is empty after processing."""
    pass


class InvalidDataFormatError(Exception):
    """Raised when the dataset has invalid format or missing columns."""
    pass


class DataLoader:
    """
    Loads and prepares product reviews dataset.
    
    This class handles loading product reviews from various sources,
    converting ratings to binary sentiment labels, and performing
    stratified train/validation/test splits.
    
    Attributes:
        dataset_name: Name of the dataset to load
        test_size: Proportion of data for test set (default 15%)
        val_size: Proportion of data for validation set (default 15%)
        random_state: Random seed for reproducibility
        train_df: Training DataFrame after loading
        val_df: Validation DataFrame after loading
        test_df: Test DataFrame after loading
    """
    
    # Minimum required reviews for a valid dataset
    MIN_REVIEWS = 3000
    IDEAL_REVIEWS = 5000
    
    # Data directory paths
    DATA_DIR = Path('data')
    RAW_DIR = DATA_DIR / 'raw'
    PROCESSED_DIR = DATA_DIR / 'processed'
    PERTURBED_DIR = DATA_DIR / 'perturbed'
    
    def __init__(
        self, 
        dataset_name: str = 'amazon_reviews', 
        test_size: float = 0.15,
        val_size: float = 0.15, 
        random_state: int = 42
    ):
        """
        Initialize the DataLoader.
        
        Args:
            dataset_name: Dataset name (e.g., 'amazon_reviews', 'amazon_polarity')
            test_size: Test set proportion (default 15%)
            val_size: Validation set proportion (default 15%)
            random_state: Seed for reproducibility
            
        Raises:
            ValueError: If split sizes are invalid
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Validate split sizes
        train_size = 1.0 - test_size - val_size
        if not (0 < train_size < 1 and 0 < test_size < 1 and 0 < val_size < 1):
            raise ValueError(
                f"Invalid split sizes: train={train_size:.2f}, "
                f"val={val_size:.2f}, test={test_size:.2f}. "
                "All sizes must be between 0 and 1."
            )
        
        # DataFrames to store splits
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        
        # Store raw data before splitting
        self._raw_df: Optional[pd.DataFrame] = None
    
    def convert_rating_to_label(self, rating: int) -> Optional[int]:
        """
        Convert numeric rating to binary label.
        
        Converts ratings 1-2 to negative (0), ratings 4-5 to positive (1),
        and rating 3 to None (neutral, to be discarded).
        
        Args:
            rating: Rating from 1 to 5
            
        Returns:
            0 (negative) for 1-2, 1 (positive) for 4-5, None for 3
            
        Validates: Requirements 1.3, 1.4
        """
        if rating in [1, 2]:
            return 0  # Negative
        elif rating in [4, 5]:
            return 1  # Positive
        elif rating == 3:
            return None  # Neutral - to be discarded
        else:
            # Invalid rating - log warning and return None
            logger.warning(f"Invalid rating value: {rating}. Expected 1-5.")
            return None
    
    def _load_amazon_reviews(self) -> pd.DataFrame:
        """
        Load Amazon product reviews dataset using Hugging Face datasets.
        
        Returns:
            DataFrame with 'text' and 'rating' columns
            
        Raises:
            DatasetNotFoundError: If dataset cannot be loaded
        """
        try:
            from datasets import load_dataset
            
            logger.info("Loading Amazon reviews dataset from Hugging Face...")
            
            # Try loading Amazon Polarity dataset (pre-labeled binary)
            # This dataset has 'content' and 'label' columns
            # Label: 0 = negative, 1 = positive
            try:
                dataset = load_dataset(
                    'amazon_polarity', 
                    split='train'
                )
                
                # Sample to get a manageable size (aim for ~10k reviews)
                # This dataset is very large, so we sample
                sample_size = min(len(dataset), 10000)
                dataset = dataset.shuffle(seed=self.random_state).select(range(sample_size))
                
                df = pd.DataFrame({
                    'text': dataset['content'],
                    # Convert 0/1 labels to ratings: 0->1 (negative), 1->5 (positive)
                    'rating': [1 if label == 0 else 5 for label in dataset['label']]
                })
                
                logger.info(f"Loaded {len(df)} reviews from Amazon Polarity dataset")
                return df
                
            except Exception as e:
                logger.warning(f"Could not load amazon_polarity: {e}")
                
            # Fallback: Try McAuley Amazon Reviews
            try:
                dataset = load_dataset(
                    'McAuley-Lab/Amazon-Reviews-2023',
                    'raw_review_All_Beauty',
                    split='full'
                )
                
                sample_size = min(len(dataset), 10000)
                dataset = dataset.shuffle(seed=self.random_state).select(range(sample_size))
                
                df = pd.DataFrame({
                    'text': dataset['text'],
                    'rating': dataset['rating']
                })
                
                logger.info(f"Loaded {len(df)} reviews from McAuley Amazon Reviews")
                return df
                
            except Exception as e:
                logger.warning(f"Could not load McAuley Amazon Reviews: {e}")
            
            # Final fallback: IMDB dataset (movie reviews, but similar task)
            try:
                dataset = load_dataset('imdb', split='train')
                
                sample_size = min(len(dataset), 10000)
                dataset = dataset.shuffle(seed=self.random_state).select(range(sample_size))
                
                df = pd.DataFrame({
                    'text': dataset['text'],
                    # Convert 0/1 labels to ratings: 0->1 (negative), 1->5 (positive)
                    'rating': [1 if label == 0 else 5 for label in dataset['label']]
                })
                
                logger.info(f"Loaded {len(df)} reviews from IMDB dataset (fallback)")
                return df
                
            except Exception as e:
                logger.warning(f"Could not load IMDB dataset: {e}")
                raise DatasetNotFoundError(
                    f"Could not load any review dataset. "
                    f"Please install the 'datasets' library: pip install datasets"
                )
                
        except ImportError:
            raise DatasetNotFoundError(
                "The 'datasets' library is required. "
                "Install it with: pip install datasets"
            )
    
    def _load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load reviews from a CSV file.
        
        Args:
            filepath: Path to CSV file with 'text' and 'rating' columns
            
        Returns:
            DataFrame with 'text' and 'rating' columns
            
        Raises:
            DatasetNotFoundError: If file not found
            InvalidDataFormatError: If required columns are missing
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise DatasetNotFoundError(f"CSV file not found: {filepath}")
        except Exception as e:
            raise DatasetNotFoundError(f"Error reading CSV file: {e}")
        
        # Check for required columns
        required_cols = {'text', 'rating'}
        if not required_cols.issubset(df.columns):
            # Try common alternative column names
            col_mapping = {
                'review': 'text',
                'review_text': 'text',
                'content': 'text',
                'body': 'text',
                'score': 'rating',
                'stars': 'rating',
                'label': 'rating'
            }
            
            for old_col, new_col in col_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Check again
            if not required_cols.issubset(df.columns):
                raise InvalidDataFormatError(
                    f"CSV must contain 'text' and 'rating' columns. "
                    f"Found columns: {list(df.columns)}"
                )
        
        return df[['text', 'rating']]
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load dataset and return train, validation, and test DataFrames.
        
        This method:
        1. Loads the raw dataset
        2. Converts ratings to binary labels (discarding neutral reviews)
        3. Performs stratified split into train/val/test sets
        
        The split is performed BEFORE any preprocessing to ensure
        no data leakage from validation/test sets to training.
        
        Returns:
            Tuple[train_df, val_df, test_df] with columns ['text', 'label']
            
        Raises:
            DatasetNotFoundError: If dataset cannot be loaded
            EmptyDatasetError: If dataset is empty after filtering
            
        Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
        """
        # Try to load from local files first (faster, no network needed)
        try:
            logger.info(f"Attempting to load local splits (seed={self.random_state})...")
            return self.load_splits(format='csv')
        except (FileNotFoundError, Exception) as e:
            logger.info(f"Local splits not found, loading from source: {e}")
        
        # Load raw data based on dataset name
        if self.dataset_name.endswith('.csv'):
            self._raw_df = self._load_from_csv(self.dataset_name)
        else:
            self._raw_df = self._load_amazon_reviews()
        
        # Validate we have data
        if self._raw_df is None or len(self._raw_df) == 0:
            raise EmptyDatasetError("Dataset is empty")
        
        logger.info(f"Raw dataset size: {len(self._raw_df)} reviews")
        
        # Extract text and rating (Requirement 1.2)
        df = self._raw_df.copy()
        
        # Remove rows with missing text or rating
        df = df.dropna(subset=['text', 'rating'])
        
        # Ensure text is string
        df['text'] = df['text'].astype(str)
        
        # Convert ratings to binary labels (Requirements 1.3, 1.4)
        df['label'] = df['rating'].apply(self.convert_rating_to_label)
        
        # Remove neutral reviews (rating 3) - Requirement 1.4
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        logger.info(f"After filtering neutral reviews: {len(df)} reviews")
        
        # Check minimum size (Requirement 1.1)
        if len(df) < self.MIN_REVIEWS:
            logger.warning(
                f"Dataset has {len(df)} reviews, below minimum of {self.MIN_REVIEWS}. "
                f"Results may not be statistically significant."
            )
        elif len(df) < self.IDEAL_REVIEWS:
            logger.info(
                f"Dataset has {len(df)} reviews. "
                f"Ideal is {self.IDEAL_REVIEWS}+ for better statistical power."
            )
        
        # Check for empty dataset
        if len(df) == 0:
            raise EmptyDatasetError(
                "No reviews remaining after filtering. "
                "Check that ratings are in range 1-5."
            )
        
        # Keep only text and label columns
        df = df[['text', 'label']]
        
        # Stratified split BEFORE preprocessing (Requirements 1.5, 1.6)
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['label'],
            random_state=self.random_state
        )
        
        # Second split: separate validation from training
        # Adjust val_size to account for already removed test set
        adjusted_val_size = self.val_size / (1 - self.test_size)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df['label'],
            random_state=self.random_state
        )
        
        # Reset indices
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        
        # Log split information
        logger.info(
            f"Data split complete:\n"
            f"  Train: {len(self.train_df)} ({len(self.train_df)/len(df)*100:.1f}%)\n"
            f"  Val:   {len(self.val_df)} ({len(self.val_df)/len(df)*100:.1f}%)\n"
            f"  Test:  {len(self.test_df)} ({len(self.test_df)/len(df)*100:.1f}%)"
        )
        
        # Report class distribution (Requirement 1.7)
        self._log_class_distribution()
        
        return self.train_df, self.val_df, self.test_df
    
    def _log_class_distribution(self) -> None:
        """Log class distribution in all sets."""
        distribution = self.get_class_distribution()
        
        logger.info("Class distribution:")
        for split_name, counts in distribution.items():
            total = counts['negative'] + counts['positive']
            neg_pct = counts['negative'] / total * 100 if total > 0 else 0
            pos_pct = counts['positive'] / total * 100 if total > 0 else 0
            logger.info(
                f"  {split_name}: negative={counts['negative']} ({neg_pct:.1f}%), "
                f"positive={counts['positive']} ({pos_pct:.1f}%)"
            )
    
    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Return class distribution in train, validation, and test sets.
        
        Returns:
            Dictionary with structure:
            {
                'train': {'negative': count, 'positive': count},
                'val': {'negative': count, 'positive': count},
                'test': {'negative': count, 'positive': count}
            }
            
        Raises:
            ValueError: If data has not been loaded yet
            
        Validates: Requirement 1.7
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError(
                "Data has not been loaded yet. Call load() first."
            )
        
        def count_labels(df: pd.DataFrame) -> Dict[str, int]:
            """Count negative and positive labels in a DataFrame."""
            counts = df['label'].value_counts()
            return {
                'negative': int(counts.get(0, 0)),
                'positive': int(counts.get(1, 0))
            }
        
        return {
            'train': count_labels(self.train_df),
            'val': count_labels(self.val_df),
            'test': count_labels(self.test_df)
        }
    
    def get_split_proportions(self) -> Dict[str, float]:
        """
        Get the actual proportions of each split.
        
        Returns:
            Dictionary with 'train', 'val', 'test' proportions
            
        Raises:
            ValueError: If data has not been loaded yet
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError(
                "Data has not been loaded yet. Call load() first."
            )
        
        total = len(self.train_df) + len(self.val_df) + len(self.test_df)
        
        return {
            'train': len(self.train_df) / total,
            'val': len(self.val_df) / total,
            'test': len(self.test_df) / total
        }
    
    def set_random_state(self, random_state: int) -> None:
        """
        Set a new random state for reproducible experiments.
        
        This allows running multiple simulations with different seeds.
        
        Args:
            random_state: New random seed
            
        Validates: Requirement 1.8
        """
        self.random_state = random_state
        # Clear cached data to force reload with new seed
        self.train_df = None
        self.val_df = None
        self.test_df = None
    
    def save_splits(self, format: str = 'csv') -> None:
        """
        Save train, validation, and test splits to disk.
        
        Saves splits to the organized folder structure:
        - data/raw/train/
        - data/raw/validation/
        - data/raw/test/
        
        Args:
            format: File format ('csv' or 'parquet')
            
        Raises:
            ValueError: If data has not been loaded yet
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Data has not been loaded yet. Call load() first.")
        
        # Create directories if they don't exist
        (self.RAW_DIR / 'train').mkdir(parents=True, exist_ok=True)
        (self.RAW_DIR / 'validation').mkdir(parents=True, exist_ok=True)
        (self.RAW_DIR / 'test').mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        if format == 'csv':
            train_path = self.RAW_DIR / 'train' / f'train_seed{self.random_state}.csv'
            val_path = self.RAW_DIR / 'validation' / f'validation_seed{self.random_state}.csv'
            test_path = self.RAW_DIR / 'test' / f'test_seed{self.random_state}.csv'
            
            # Save as CSV
            self.train_df.to_csv(train_path, index=False)
            self.val_df.to_csv(val_path, index=False)
            self.test_df.to_csv(test_path, index=False)
        elif format == 'parquet':
            train_path = self.RAW_DIR / 'train' / f'train_seed{self.random_state}.parquet'
            val_path = self.RAW_DIR / 'validation' / f'validation_seed{self.random_state}.parquet'
            test_path = self.RAW_DIR / 'test' / f'test_seed{self.random_state}.parquet'
            
            # Save as Parquet
            self.train_df.to_parquet(train_path, index=False)
            self.val_df.to_parquet(val_path, index=False)
            self.test_df.to_parquet(test_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.")
        
        logger.info(f"Saved splits to {self.RAW_DIR} (seed={self.random_state})")
    
    def load_splits(self, format: str = 'csv') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test splits from disk.
        
        Loads splits from the organized folder structure:
        - data/raw/train/
        - data/raw/validation/
        - data/raw/test/
        
        Args:
            format: File format ('csv' or 'parquet')
            
        Returns:
            Tuple[train_df, val_df, test_df] with columns ['text', 'label']
            
        Raises:
            FileNotFoundError: If split files don't exist
        """
        # Define file paths
        if format == 'csv':
            train_path = self.RAW_DIR / 'train' / f'train_seed{self.random_state}.csv'
            val_path = self.RAW_DIR / 'validation' / f'validation_seed{self.random_state}.csv'
            test_path = self.RAW_DIR / 'test' / f'test_seed{self.random_state}.csv'
            
            # Load from CSV
            self.train_df = pd.read_csv(train_path)
            self.val_df = pd.read_csv(val_path)
            self.test_df = pd.read_csv(test_path)
        elif format == 'parquet':
            train_path = self.RAW_DIR / 'train' / f'train_seed{self.random_state}.parquet'
            val_path = self.RAW_DIR / 'validation' / f'validation_seed{self.random_state}.parquet'
            test_path = self.RAW_DIR / 'test' / f'test_seed{self.random_state}.parquet'
            
            # Load from Parquet
            self.train_df = pd.read_parquet(train_path)
            self.val_df = pd.read_parquet(val_path)
            self.test_df = pd.read_parquet(test_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.")
        
        logger.info(f"Loaded splits from {self.RAW_DIR} (seed={self.random_state})")
        
        return self.train_df, self.val_df, self.test_df
    
    def save_processed_splits(self, train_processed: pd.DataFrame, 
                             val_processed: pd.DataFrame, 
                             test_processed: pd.DataFrame,
                             suffix: str = '',
                             format: str = 'csv') -> None:
        """
        Save processed train, validation, and test splits to disk.
        
        Saves processed splits to:
        - data/processed/train/
        - data/processed/validation/
        - data/processed/test/
        
        Args:
            train_processed: Processed training DataFrame
            val_processed: Processed validation DataFrame
            test_processed: Processed test DataFrame
            suffix: Optional suffix for filename (e.g., 'bow', 'embeddings')
            format: File format ('csv' or 'parquet')
        """
        # Create directories if they don't exist
        (self.PROCESSED_DIR / 'train').mkdir(parents=True, exist_ok=True)
        (self.PROCESSED_DIR / 'validation').mkdir(parents=True, exist_ok=True)
        (self.PROCESSED_DIR / 'test').mkdir(parents=True, exist_ok=True)
        
        # Build filename
        suffix_str = f'_{suffix}' if suffix else ''
        
        # Define file paths
        if format == 'csv':
            train_path = self.PROCESSED_DIR / 'train' / f'train{suffix_str}_seed{self.random_state}.csv'
            val_path = self.PROCESSED_DIR / 'validation' / f'validation{suffix_str}_seed{self.random_state}.csv'
            test_path = self.PROCESSED_DIR / 'test' / f'test{suffix_str}_seed{self.random_state}.csv'
            
            # Save as CSV
            train_processed.to_csv(train_path, index=False)
            val_processed.to_csv(val_path, index=False)
            test_processed.to_csv(test_path, index=False)
        elif format == 'parquet':
            train_path = self.PROCESSED_DIR / 'train' / f'train{suffix_str}_seed{self.random_state}.parquet'
            val_path = self.PROCESSED_DIR / 'validation' / f'validation{suffix_str}_seed{self.random_state}.parquet'
            test_path = self.PROCESSED_DIR / 'test' / f'test{suffix_str}_seed{self.random_state}.parquet'
            
            # Save as Parquet
            train_processed.to_parquet(train_path, index=False)
            val_processed.to_parquet(val_path, index=False)
            test_processed.to_parquet(test_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.")
        
        logger.info(f"Saved processed splits to {self.PROCESSED_DIR} (seed={self.random_state}, suffix='{suffix}')")
    
    def get_sample_reviews(self, n: int = 5) -> Dict[str, List[Dict]]:
        """
        Get sample reviews from each split for inspection.
        
        Args:
            n: Number of samples per split
            
        Returns:
            Dictionary with sample reviews from train, val, test
        """
        if self.train_df is None:
            raise ValueError("Data has not been loaded yet. Call load() first.")
        
        def sample_from_df(df: pd.DataFrame, n: int) -> List[Dict]:
            samples = df.sample(min(n, len(df)), random_state=self.random_state)
            return samples.to_dict('records')
        
        return {
            'train': sample_from_df(self.train_df, n),
            'val': sample_from_df(self.val_df, n),
            'test': sample_from_df(self.test_df, n)
        }
    
    def save_perturbed_test(self, perturbed_df: pd.DataFrame, 
                           perturbation_type: str = 'typos',
                           format: str = 'csv') -> None:
        """
        Save perturbed test data to disk.
        
        Saves perturbed test data to:
        - data/perturbed/test/
        
        Args:
            perturbed_df: Perturbed test DataFrame
            perturbation_type: Type of perturbation (e.g., 'typos', 'emojis')
            format: File format ('csv' or 'parquet')
        """
        # Create directory if it doesn't exist
        (self.PERTURBED_DIR / 'test').mkdir(parents=True, exist_ok=True)
        
        # Define file path
        if format == 'csv':
            path = self.PERTURBED_DIR / 'test' / f'test_{perturbation_type}_seed{self.random_state}.csv'
            perturbed_df.to_csv(path, index=False)
        elif format == 'parquet':
            path = self.PERTURBED_DIR / 'test' / f'test_{perturbation_type}_seed{self.random_state}.parquet'
            perturbed_df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'parquet'.")
        
        logger.info(f"Saved perturbed test data to {path}")

