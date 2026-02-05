"""
Embedding encoding module for sentiment analysis NLP project.

This module provides the EmbeddingEncoder class for converting texts to
embedding vectors using pre-trained word embeddings (Word2Vec or GloVe).
It calculates the mean vector of word embeddings for document representation.

"""

from typing import List, Optional
import logging
import numpy as np
import gensim.downloader as api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Encodes texts using pre-trained word embeddings.
    
    This class loads pre-trained Word2Vec or GloVe embeddings and converts
    texts to dense vectors by calculating the mean of word embeddings.
    Out-of-vocabulary (OOV) words are handled by using zero vectors.
    
    Key features:
    - Loads pre-trained Word2Vec or GloVe embeddings (Requirement 4.1)
    - Calculates mean vector of word embeddings for documents (Requirement 4.2)
    - Handles OOV words by ignoring them or using zero vector (Requirement 4.3)
    
    Attributes:
        model_name: Name of the embedding model (e.g., 'glove-wiki-gigaword-100')
        model: The loaded gensim KeyedVectors model
        embedding_dim: Dimensionality of the embedding vectors
    """
    
    def __init__(self, model_name: str = 'glove-wiki-gigaword-100'):
        """
        Initialize the EmbeddingEncoder.
        
        Args:
            model_name: Embedding model name from gensim-data
                       (default 'glove-wiki-gigaword-100')
                       
        Common models:
            - 'glove-wiki-gigaword-50': GloVe 50-dim (66MB)
            - 'glove-wiki-gigaword-100': GloVe 100-dim (128MB)
            - 'glove-wiki-gigaword-200': GloVe 200-dim (252MB)
            - 'glove-wiki-gigaword-300': GloVe 300-dim (376MB)
            - 'word2vec-google-news-300': Word2Vec 300-dim (1.6GB)
            
        Validates: Requirement 4.1 - Load pre-trained Word2Vec or GloVe embeddings
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
        logger.info(f"Loading embedding model: {model_name}...")
        
        try:
            # Load pre-trained embeddings from gensim-data
            self.model = api.load(model_name)
            self.embedding_dim = self.model.vector_size
            
            logger.info(
                f"Embedding model loaded successfully. "
                f"Vocabulary size: {len(self.model)}, "
                f"Embedding dimension: {self.embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'. "
                f"Make sure the model name is valid and you have internet connection "
                f"for the first download."
            ) from e
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text as mean vector of word embeddings.
        
        This method tokenizes the text (splits on whitespace), looks up
        each word in the embedding model, and calculates the mean of all
        word vectors. Words not in the vocabulary (OOV) are ignored.
        
        If no words are found in the vocabulary, returns a zero vector.
        
        Args:
            text: Preprocessed text (space-separated tokens)
            
        Returns:
            Mean embedding vector of shape (embedding_dim,)
            
        Validates:
            Requirement 4.2 - Calculate mean vector of embeddings for all words
            Requirement 4.3 - Handle OOV words by ignoring them or using zero vector
        """
        # Handle None or empty text
        if not text or not isinstance(text, str):
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Tokenize text (split on whitespace)
        tokens = text.split()
        
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Collect embeddings for words in vocabulary
        embeddings = []
        oov_count = 0
        
        for token in tokens:
            if token in self.model:
                embeddings.append(self.model[token])
            else:
                oov_count += 1
        
        # If no words found in vocabulary, return zero vector (Requirement 4.3)
        if not embeddings:
            logger.debug(
                f"All {len(tokens)} tokens are OOV. Returning zero vector."
            )
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Calculate mean vector (Requirement 4.2)
        mean_vector = np.mean(embeddings, axis=0).astype(np.float32)
        
        # Log OOV statistics if significant
        if oov_count > 0 and len(tokens) > 0:
            oov_rate = oov_count / len(tokens)
            if oov_rate > 0.3:  # Log if more than 30% OOV
                logger.debug(
                    f"High OOV rate: {oov_rate:.1%} ({oov_count}/{len(tokens)} tokens)"
                )
        
        return mean_vector
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode batch of texts.
        
        Convenience method to encode multiple texts at once.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Array of embedding vectors with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        logger.info(f"Encoding batch of {len(texts)} texts...")
        
        # Encode each text
        embeddings = [self.encode(text) for text in texts]
        
        # Stack into array
        embeddings_array = np.vstack(embeddings)
        
        logger.info(f"Batch encoding complete. Shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the embedding vocabulary.
        
        Returns:
            Number of words in the embedding model
        """
        return len(self.model)
    
    def is_in_vocabulary(self, word: str) -> bool:
        """
        Check if a word is in the embedding vocabulary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is in vocabulary, False otherwise
        """
        return word in self.model
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a specific word.
        
        Args:
            word: Word to get embedding for
            
        Returns:
            Embedding vector if word is in vocabulary, None otherwise
        """
        if word in self.model:
            return self.model[word]
        return None
    
    def get_oov_rate(self, texts: List[str]) -> float:
        """
        Calculate the out-of-vocabulary rate for a list of texts.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            OOV rate as a float between 0 and 1
        """
        total_tokens = 0
        oov_tokens = 0
        
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            
            tokens = text.split()
            total_tokens += len(tokens)
            
            for token in tokens:
                if token not in self.model:
                    oov_tokens += 1
        
        if total_tokens == 0:
            return 0.0
        
        return oov_tokens / total_tokens
    
    def get_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        """
        Get the most similar words to a given word.
        
        Args:
            word: Word to find similar words for
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
            
        Raises:
            KeyError: If word is not in vocabulary
        """
        if word not in self.model:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        return self.model.most_similar(word, topn=topn)
