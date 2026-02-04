"""
Text preprocessing module for sentiment analysis NLP project.

This module provides the Preprocessor class for cleaning and normalizing texts
for traditional SVM models. It follows the fit/transform pattern to prevent
data leakage by fitting only on training data.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

import re
import string
from typing import List, Optional, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocesses texts for traditional models (SVM).
    
    This class implements text preprocessing with a fit/transform pattern
    to ensure no data leakage from validation/test sets to training.
    The preprocessor is fitted ONLY on training data, and the same
    transformations are applied to all sets.
    
    Preprocessing steps:
    1. Convert to lowercase (Requirement 2.2)
    2. Remove special characters and excessive punctuation (Requirement 2.3)
    3. Remove stopwords (Requirement 2.4)
    4. Tokenize text (Requirement 2.5)
    
    Attributes:
        language: Language for stopwords (default 'english')
        remove_stopwords: Whether to remove stopwords
        stopwords: Set of stopwords for the specified language
        is_fitted: Whether the preprocessor has been fitted
        vocabulary: Set of words seen during training (for OOV handling)
    """
    
    # Default English stopwords (NLTK-based list)
    # This is a predefined list to avoid NLTK download requirements
    ENGLISH_STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
        'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
        'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
        "won't", 'wouldn', "wouldn't"
    }
    
    def __init__(self, language: str = 'english', remove_stopwords: bool = True):
        """
        Initialize the Preprocessor.
        
        Args:
            language: Language for stopwords (default 'english')
            remove_stopwords: Whether to remove stopwords (default True)
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        
        # Initialize stopwords based on language
        self.stopwords: Set[str] = self._load_stopwords(language)
        
        # Track fitting state (Requirement 2.1)
        self.is_fitted: bool = False
        
        # Vocabulary learned from training (for OOV handling - Requirement 2.7)
        self.vocabulary: Set[str] = set()
        
        # Compile regex patterns for efficiency
        # Pattern to remove special characters but keep alphanumeric and spaces
        self._special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        # Pattern to remove excessive whitespace
        self._whitespace_pattern = re.compile(r'\s+')
    
    def _load_stopwords(self, language: str) -> Set[str]:
        """
        Load stopwords for the specified language.
        
        Args:
            language: Language code (e.g., 'english')
            
        Returns:
            Set of stopwords
        """
        if language.lower() == 'english':
            return self.ENGLISH_STOPWORDS.copy()
        else:
            # Try to load from NLTK if available
            try:
                import nltk
                try:
                    stopwords_list = nltk.corpus.stopwords.words(language)
                    return set(stopwords_list)
                except LookupError:
                    # Download stopwords if not available
                    nltk.download('stopwords', quiet=True)
                    stopwords_list = nltk.corpus.stopwords.words(language)
                    return set(stopwords_list)
            except ImportError:
                logger.warning(
                    f"NLTK not available for language '{language}'. "
                    f"Using English stopwords as fallback."
                )
                return self.ENGLISH_STOPWORDS.copy()
    
    def _to_lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Input text
            
        Returns:
            Lowercase text
            
        Validates: Requirement 2.2
        """
        return text.lower()
    
    def _remove_special_characters(self, text: str) -> str:
        """
        Remove special characters and excessive punctuation.
        
        Keeps only alphanumeric characters and spaces.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
            
        Validates: Requirement 2.3
        """
        # Remove special characters
        text = self._special_char_pattern.sub(' ', text)
        # Normalize whitespace
        text = self._whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Simple whitespace tokenization after preprocessing.
        
        Args:
            text: Input text (should be preprocessed)
            
        Returns:
            List of tokens
            
        Validates: Requirement 2.5
        """
        return text.split()
    
    def _remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
            
        Validates: Requirement 2.4
        """
        if not self.remove_stopwords:
            return tokens
        return [token for token in tokens if token not in self.stopwords]
    
    def _preprocess_single(self, text: str) -> str:
        """
        Apply all preprocessing steps to a single text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text as space-separated tokens
        """
        # Handle None or non-string input
        if text is None:
            return ''
        if not isinstance(text, str):
            text = str(text)
        
        # Handle empty text
        if not text.strip():
            return ''
        
        # Step 1: Lowercase (Requirement 2.2)
        text = self._to_lowercase(text)
        
        # Step 2: Remove special characters (Requirement 2.3)
        text = self._remove_special_characters(text)
        
        # Step 3: Tokenize (Requirement 2.5)
        tokens = self._tokenize(text)
        
        # Step 4: Remove stopwords (Requirement 2.4)
        tokens = self._remove_stopwords_from_tokens(tokens)
        
        # Return as space-separated string
        return ' '.join(tokens)
    
    def fit(self, texts: List[str]) -> 'Preprocessor':
        """
        Fit preprocessor on training set.
        
        This method learns the vocabulary from the training set,
        which is used for consistent OOV handling during transform.
        The preprocessor should be fitted ONLY on training data
        to avoid data leakage.
        
        Args:
            texts: List of training texts
            
        Returns:
            self for method chaining
            
        Validates: Requirements 2.1, 2.6
        """
        logger.info(f"Fitting preprocessor on {len(texts)} training texts...")
        
        # Reset vocabulary
        self.vocabulary = set()
        
        # Process each text and build vocabulary
        for text in texts:
            preprocessed = self._preprocess_single(text)
            tokens = preprocessed.split()
            self.vocabulary.update(tokens)
        
        self.is_fitted = True
        
        logger.info(f"Preprocessor fitted. Vocabulary size: {len(self.vocabulary)}")
        
        return self
    
    def transform(self, texts: List[str]) -> List[str]:
        """
        Transform texts using learned parameters.
        
        Applies the same preprocessing transformations to all texts.
        When processing validation or test sets, uses ONLY the
        transformations learned from training (Requirement 2.6).
        
        Unknown words (OOV) are handled consistently by keeping them
        in the output (Requirement 2.7). The vocabulary is only used
        for tracking, not for filtering.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            List of preprocessed texts
            
        Raises:
            RuntimeError: If transform is called before fit
            
        Validates: Requirements 2.6, 2.7
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. Call fit() first."
            )
        
        preprocessed_texts = []
        oov_count = 0
        total_tokens = 0
        
        for text in texts:
            preprocessed = self._preprocess_single(text)
            preprocessed_texts.append(preprocessed)
            
            # Track OOV statistics (for logging)
            tokens = preprocessed.split()
            total_tokens += len(tokens)
            oov_count += sum(1 for t in tokens if t not in self.vocabulary)
        
        # Log OOV statistics
        if total_tokens > 0:
            oov_rate = oov_count / total_tokens * 100
            if oov_rate > 10:
                logger.info(
                    f"OOV rate: {oov_rate:.1f}% ({oov_count}/{total_tokens} tokens)"
                )
        
        return preprocessed_texts
    
    def fit_transform(self, texts: List[str]) -> List[str]:
        """
        Fit and transform in one step.
        
        Convenience method that calls fit() then transform().
        Should only be used on training data.
        
        Args:
            texts: List of training texts
            
        Returns:
            List of preprocessed texts
        """
        return self.fit(texts).transform(texts)
    
    def get_stopwords(self) -> Set[str]:
        """
        Get the current stopwords set.
        
        Returns:
            Set of stopwords being used
        """
        return self.stopwords.copy()
    
    def add_stopwords(self, words: List[str]) -> None:
        """
        Add custom stopwords to the existing set.
        
        Args:
            words: List of words to add as stopwords
        """
        self.stopwords.update(word.lower() for word in words)
    
    def remove_from_stopwords(self, words: List[str]) -> None:
        """
        Remove words from the stopwords set.
        
        Args:
            words: List of words to remove from stopwords
        """
        for word in words:
            self.stopwords.discard(word.lower())
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the learned vocabulary.
        
        Returns:
            Number of unique tokens in vocabulary
            
        Raises:
            RuntimeError: If preprocessor has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. Call fit() first."
            )
        return len(self.vocabulary)
    
    def is_oov(self, word: str) -> bool:
        """
        Check if a word is out-of-vocabulary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is not in vocabulary, False otherwise
            
        Raises:
            RuntimeError: If preprocessor has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. Call fit() first."
            )
        return word.lower() not in self.vocabulary


# Note: BERT uses its own tokenizer and should NOT use this preprocessor
# (Requirement 2.8). The BERTClassifier class will handle its own tokenization.
