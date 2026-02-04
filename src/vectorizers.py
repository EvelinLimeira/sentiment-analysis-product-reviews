"""
TF-IDF vectorization module for sentiment analysis NLP project.

This module provides the BoWVectorizer class for converting preprocessed texts
to TF-IDF vectors. It follows the fit/transform pattern to prevent data leakage
by fitting only on training data.

Requirements: 3.1, 3.2, 3.3
"""

from typing import List, Tuple, Optional
import logging
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoWVectorizer:
    """
    Vectorizes texts using TF-IDF (Term Frequency-Inverse Document Frequency).
    
    This class wraps sklearn's TfidfVectorizer and implements a fit/transform
    pattern to ensure no data leakage from validation/test sets to training.
    The vectorizer is fitted ONLY on training data, and the same vocabulary
    is used to transform all sets.
    
    Key features:
    - Fitted ONLY on training set (Requirement 3.1)
    - Converts texts to TF-IDF vectors with configurable max_features and ngram_range (Requirement 3.2)
    - Uses ONLY vocabulary learned from training when transforming validation/test sets (Requirement 3.3)
    
    Attributes:
        max_features: Maximum number of features (vocabulary size)
        ngram_range: N-gram range for feature extraction (e.g., (1, 2) for unigrams and bigrams)
        is_fitted: Whether the vectorizer has been fitted
        vectorizer: The underlying sklearn TfidfVectorizer
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the BoWVectorizer.
        
        Args:
            max_features: Maximum number of features (default 5000)
            ngram_range: N-gram range (default (1, 2) for unigrams and bigrams)
            
        Raises:
            ValueError: If max_features < 1 or ngram_range is invalid
        """
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        
        if not isinstance(ngram_range, tuple) or len(ngram_range) != 2:
            raise ValueError(f"ngram_range must be a tuple of 2 integers, got {ngram_range}")
        
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError(
                f"ngram_range must have min >= 1 and max >= min, got {ngram_range}"
            )
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.is_fitted = False
        
        # Initialize the underlying TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            # Use default tokenizer (splits on whitespace for preprocessed text)
            lowercase=False,  # Text should already be lowercased by Preprocessor
            token_pattern=r'(?u)\b\w+\b',  # Match word tokens
        )
    
    def fit(self, texts: List[str]) -> 'BoWVectorizer':
        """
        Fit vocabulary on training set.
        
        This method learns the vocabulary from the training set.
        The vectorizer should be fitted ONLY on training data
        to avoid data leakage.
        
        Args:
            texts: List of preprocessed training texts
            
        Returns:
            self for method chaining
            
        Validates: Requirement 3.1 - BoW_Vectorizer SHALL be fitted ONLY on the training set
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
        
        logger.info(f"Fitting BoWVectorizer on {len(texts)} training texts...")
        
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(
            f"BoWVectorizer fitted. Vocabulary size: {vocab_size} "
            f"(max_features={self.max_features}, ngram_range={self.ngram_range})"
        )
        
        return self
    
    def transform(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        """
        Transform texts to sparse TF-IDF matrix.
        
        Uses ONLY the vocabulary learned from training. Words not in the
        training vocabulary are ignored (assigned zero weight).
        
        Args:
            texts: List of preprocessed texts to transform
            
        Returns:
            Sparse CSR matrix of TF-IDF vectors with shape (n_texts, n_features)
            
        Raises:
            RuntimeError: If transform is called before fit
            
        Validates: 
            Requirement 3.2 - Convert preprocessed texts to TF-IDF vectors
            Requirement 3.3 - Use ONLY vocabulary learned from training
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BoWVectorizer has not been fitted. Call fit() first."
            )
        
        # Transform texts using the fitted vocabulary
        # Words not in vocabulary are automatically ignored by sklearn
        tfidf_matrix = self.vectorizer.transform(texts)
        
        logger.debug(
            f"Transformed {len(texts)} texts to TF-IDF matrix "
            f"with shape {tfidf_matrix.shape}"
        )
        
        return tfidf_matrix
    
    def fit_transform(self, texts: List[str]) -> scipy.sparse.csr_matrix:
        """
        Fit and transform in one step.
        
        Convenience method that calls fit() then transform().
        Should only be used on training data.
        
        Args:
            texts: List of preprocessed training texts
            
        Returns:
            Sparse CSR matrix of TF-IDF vectors
        """
        return self.fit(texts).transform(texts)
    
    def get_vocabulary(self) -> dict:
        """
        Get the learned vocabulary.
        
        Returns:
            Dictionary mapping terms to feature indices
            
        Raises:
            RuntimeError: If vectorizer has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BoWVectorizer has not been fitted. Call fit() first."
            )
        return self.vectorizer.vocabulary_.copy()
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the learned vocabulary.
        
        Returns:
            Number of features (vocabulary size)
            
        Raises:
            RuntimeError: If vectorizer has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BoWVectorizer has not been fitted. Call fit() first."
            )
        return len(self.vectorizer.vocabulary_)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary terms).
        
        Returns:
            List of feature names in order of their indices
            
        Raises:
            RuntimeError: If vectorizer has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BoWVectorizer has not been fitted. Call fit() first."
            )
        return self.vectorizer.get_feature_names_out().tolist()
    
    def is_in_vocabulary(self, term: str) -> bool:
        """
        Check if a term is in the vocabulary.
        
        Args:
            term: Term to check
            
        Returns:
            True if term is in vocabulary, False otherwise
            
        Raises:
            RuntimeError: If vectorizer has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BoWVectorizer has not been fitted. Call fit() first."
            )
        return term in self.vectorizer.vocabulary_
    
    def get_idf(self, term: str) -> Optional[float]:
        """
        Get the IDF (Inverse Document Frequency) value for a term.
        
        Args:
            term: Term to get IDF for
            
        Returns:
            IDF value if term is in vocabulary, None otherwise
            
        Raises:
            RuntimeError: If vectorizer has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BoWVectorizer has not been fitted. Call fit() first."
            )
        
        if term not in self.vectorizer.vocabulary_:
            return None
        
        idx = self.vectorizer.vocabulary_[term]
        return self.vectorizer.idf_[idx]
