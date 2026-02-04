"""
Unit tests for the EmbeddingEncoder class.

Tests cover:
- Initialization and model loading (Requirement 4.1)
- Mean vector calculation (Requirement 4.2)
- OOV word handling (Requirement 4.3)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.embedding_encoder import EmbeddingEncoder


class TestEmbeddingEncoderInit:
    """Tests for EmbeddingEncoder initialization."""
    
    @patch('src.embedding_encoder.api.load')
    def test_default_model_name(self, mock_load):
        """Test default model name is glove-wiki-gigaword-100."""
        mock_model = Mock()
        mock_model.vector_size = 100
        mock_model.__len__ = Mock(return_value=400000)
        mock_load.return_value = mock_model
        
        encoder = EmbeddingEncoder()
        
        assert encoder.model_name == 'glove-wiki-gigaword-100'
        mock_load.assert_called_once_with('glove-wiki-gigaword-100')
    
    @patch('src.embedding_encoder.api.load')
    def test_custom_model_name(self, mock_load):
        """Test custom model name."""
        mock_model = Mock()
        mock_model.vector_size = 50
        mock_model.__len__ = Mock(return_value=400000)
        mock_load.return_value = mock_model
        
        encoder = EmbeddingEncoder(model_name='glove-wiki-gigaword-50')
        
        assert encoder.model_name == 'glove-wiki-gigaword-50'
        mock_load.assert_called_once_with('glove-wiki-gigaword-50')
    
    @patch('src.embedding_encoder.api.load')
    def test_model_loaded_successfully(self, mock_load):
        """Test that model is loaded and attributes are set."""
        mock_model = Mock()
        mock_model.vector_size = 100
        mock_model.__len__ = Mock(return_value=400000)
        mock_load.return_value = mock_model
        
        encoder = EmbeddingEncoder()
        
        assert encoder.model is not None
        assert encoder.embedding_dim == 100
    
    @patch('src.embedding_encoder.api.load')
    def test_model_load_failure_raises_error(self, mock_load):
        """Test that model load failure raises RuntimeError."""
        mock_load.side_effect = Exception("Network error")
        
        with pytest.raises(RuntimeError, match="Could not load embedding model"):
            EmbeddingEncoder()


class TestEmbeddingEncoderEncode:
    """Tests for encode() method (Requirements 4.2, 4.3)."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder with controlled vocabulary."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            # Create mock model with known embeddings
            mock_model = Mock()
            mock_model.vector_size = 3
            mock_model.__len__ = Mock(return_value=3)
            
            # Define embeddings for known words
            embeddings = {
                'hello': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'world': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'test': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
            
            # Mock __contains__ and __getitem__
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_encode_single_word(self, mock_encoder):
        """Test encoding a single word."""
        result = mock_encoder.encode('hello')
        
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_multiple_words_mean_vector(self, mock_encoder):
        """Test that encode calculates mean vector (Requirement 4.2)."""
        # 'hello' = [1, 0, 0], 'world' = [0, 1, 0]
        # Mean = [0.5, 0.5, 0]
        result = mock_encoder.encode('hello world')
        
        expected = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_three_words_mean_vector(self, mock_encoder):
        """Test mean vector calculation with three words."""
        # 'hello' = [1, 0, 0], 'world' = [0, 1, 0], 'test' = [0, 0, 1]
        # Mean = [1/3, 1/3, 1/3]
        result = mock_encoder.encode('hello world test')
        
        expected = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_empty_text_returns_zero_vector(self, mock_encoder):
        """Test that empty text returns zero vector (Requirement 4.3)."""
        result = mock_encoder.encode('')
        
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_none_returns_zero_vector(self, mock_encoder):
        """Test that None returns zero vector."""
        result = mock_encoder.encode(None)
        
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_whitespace_only_returns_zero_vector(self, mock_encoder):
        """Test that whitespace-only text returns zero vector."""
        result = mock_encoder.encode('   ')
        
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_all_oov_words_returns_zero_vector(self, mock_encoder):
        """Test that all OOV words return zero vector (Requirement 4.3)."""
        # 'unknown' and 'words' are not in vocabulary
        result = mock_encoder.encode('unknown words')
        
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_mixed_known_and_oov_words(self, mock_encoder):
        """Test that OOV words are ignored in mean calculation (Requirement 4.3)."""
        # 'hello' = [1, 0, 0], 'unknown' is OOV, 'world' = [0, 1, 0]
        # Mean should be calculated only from 'hello' and 'world'
        # Mean = [0.5, 0.5, 0]
        result = mock_encoder.encode('hello unknown world')
        
        expected = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_returns_correct_dimension(self, mock_encoder):
        """Test that encoded vector has correct dimension."""
        result = mock_encoder.encode('hello world')
        
        assert result.shape == (3,)
        assert len(result) == mock_encoder.embedding_dim
    
    def test_encode_returns_float32(self, mock_encoder):
        """Test that encoded vector is float32."""
        result = mock_encoder.encode('hello world')
        
        assert result.dtype == np.float32
    
    def test_encode_non_string_input(self, mock_encoder):
        """Test that non-string input is handled gracefully."""
        result = mock_encoder.encode(123)
        
        # Should return zero vector for non-string
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestEmbeddingEncoderEncodeBatch:
    """Tests for encode_batch() method."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder with controlled vocabulary."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 3
            mock_model.__len__ = Mock(return_value=3)
            
            embeddings = {
                'hello': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'world': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'test': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_encode_batch_empty_list(self, mock_encoder):
        """Test encoding empty list."""
        result = mock_encoder.encode_batch([])
        
        assert result.shape == (0, 3)
    
    def test_encode_batch_single_text(self, mock_encoder):
        """Test encoding batch with single text."""
        result = mock_encoder.encode_batch(['hello world'])
        
        assert result.shape == (1, 3)
        expected = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_batch_multiple_texts(self, mock_encoder):
        """Test encoding batch with multiple texts."""
        texts = ['hello', 'world', 'test']
        result = mock_encoder.encode_batch(texts)
        
        assert result.shape == (3, 3)
        
        expected = np.array([
            [1.0, 0.0, 0.0],  # hello
            [0.0, 1.0, 0.0],  # world
            [0.0, 0.0, 1.0],  # test
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_batch_returns_correct_shape(self, mock_encoder):
        """Test that batch encoding returns correct shape."""
        texts = ['hello world', 'test', 'hello test']
        result = mock_encoder.encode_batch(texts)
        
        assert result.shape == (3, 3)
        assert result.shape[0] == len(texts)
        assert result.shape[1] == mock_encoder.embedding_dim
    
    def test_encode_batch_with_empty_texts(self, mock_encoder):
        """Test batch encoding with some empty texts."""
        texts = ['hello', '', 'world']
        result = mock_encoder.encode_batch(texts)
        
        assert result.shape == (3, 3)
        
        # Second row should be zero vector
        np.testing.assert_array_equal(result[1], np.zeros(3, dtype=np.float32))
    
    def test_encode_batch_consistent_with_encode(self, mock_encoder):
        """Test that encode_batch produces same results as individual encode calls."""
        texts = ['hello world', 'test', 'hello test']
        
        batch_result = mock_encoder.encode_batch(texts)
        individual_results = np.array([mock_encoder.encode(text) for text in texts])
        
        np.testing.assert_array_almost_equal(batch_result, individual_results)


class TestEmbeddingEncoderVocabularyMethods:
    """Tests for vocabulary-related methods."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder with controlled vocabulary."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 100
            mock_model.__len__ = Mock(return_value=3)
            
            embeddings = {
                'hello': np.array([1.0] * 100, dtype=np.float32),
                'world': np.array([0.5] * 100, dtype=np.float32),
                'test': np.array([0.0] * 100, dtype=np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_get_embedding_dim(self, mock_encoder):
        """Test get_embedding_dim returns correct dimension."""
        assert mock_encoder.get_embedding_dim() == 100
    
    def test_get_vocabulary_size(self, mock_encoder):
        """Test get_vocabulary_size returns correct size."""
        assert mock_encoder.get_vocabulary_size() == 3
    
    def test_is_in_vocabulary_known_word(self, mock_encoder):
        """Test is_in_vocabulary for known word."""
        assert mock_encoder.is_in_vocabulary('hello') is True
        assert mock_encoder.is_in_vocabulary('world') is True
        assert mock_encoder.is_in_vocabulary('test') is True
    
    def test_is_in_vocabulary_unknown_word(self, mock_encoder):
        """Test is_in_vocabulary for unknown word."""
        assert mock_encoder.is_in_vocabulary('unknown') is False
        assert mock_encoder.is_in_vocabulary('oov') is False
    
    def test_get_word_vector_known_word(self, mock_encoder):
        """Test get_word_vector for known word."""
        vector = mock_encoder.get_word_vector('hello')
        
        assert vector is not None
        assert len(vector) == 100
        assert isinstance(vector, np.ndarray)
    
    def test_get_word_vector_unknown_word(self, mock_encoder):
        """Test get_word_vector for unknown word returns None."""
        vector = mock_encoder.get_word_vector('unknown')
        
        assert vector is None


class TestEmbeddingEncoderOOVRate:
    """Tests for OOV rate calculation."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder with controlled vocabulary."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 3
            mock_model.__len__ = Mock(return_value=3)
            
            embeddings = {
                'hello': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'world': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'test': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_get_oov_rate_no_oov(self, mock_encoder):
        """Test OOV rate when all words are in vocabulary."""
        texts = ['hello world', 'test hello']
        oov_rate = mock_encoder.get_oov_rate(texts)
        
        assert oov_rate == 0.0
    
    def test_get_oov_rate_all_oov(self, mock_encoder):
        """Test OOV rate when all words are OOV."""
        texts = ['unknown words', 'oov tokens']
        oov_rate = mock_encoder.get_oov_rate(texts)
        
        assert oov_rate == 1.0
    
    def test_get_oov_rate_mixed(self, mock_encoder):
        """Test OOV rate with mixed known and OOV words."""
        # 'hello' and 'world' are known, 'unknown' and 'oov' are not
        # Total: 4 tokens, 2 OOV -> 50%
        texts = ['hello unknown', 'world oov']
        oov_rate = mock_encoder.get_oov_rate(texts)
        
        assert oov_rate == 0.5
    
    def test_get_oov_rate_empty_list(self, mock_encoder):
        """Test OOV rate with empty list."""
        oov_rate = mock_encoder.get_oov_rate([])
        
        assert oov_rate == 0.0
    
    def test_get_oov_rate_empty_texts(self, mock_encoder):
        """Test OOV rate with empty texts."""
        texts = ['', '   ', '']
        oov_rate = mock_encoder.get_oov_rate(texts)
        
        assert oov_rate == 0.0
    
    def test_get_oov_rate_none_texts(self, mock_encoder):
        """Test OOV rate with None texts."""
        texts = [None, 'hello', None]
        oov_rate = mock_encoder.get_oov_rate(texts)
        
        # Only 'hello' is counted, 0 OOV out of 1 token
        assert oov_rate == 0.0


class TestEmbeddingEncoderSimilarWords:
    """Tests for get_similar_words method."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder with most_similar support."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 100
            mock_model.__len__ = Mock(return_value=3)
            
            embeddings = {
                'hello': np.array([1.0] * 100, dtype=np.float32),
                'world': np.array([0.9] * 100, dtype=np.float32),
                'test': np.array([0.1] * 100, dtype=np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            # Mock most_similar
            def mock_most_similar(word, topn=10):
                if word == 'hello':
                    return [('world', 0.95), ('test', 0.5)]
                return []
            
            mock_model.most_similar = mock_most_similar
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_get_similar_words_known_word(self, mock_encoder):
        """Test get_similar_words for known word."""
        similar = mock_encoder.get_similar_words('hello', topn=2)
        
        assert len(similar) == 2
        assert similar[0] == ('world', 0.95)
        assert similar[1] == ('test', 0.5)
    
    def test_get_similar_words_unknown_word_raises_error(self, mock_encoder):
        """Test get_similar_words for unknown word raises KeyError."""
        with pytest.raises(KeyError, match="not in vocabulary"):
            mock_encoder.get_similar_words('unknown')


class TestEmbeddingEncoderIntegration:
    """Integration tests with real embeddings (if available)."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Requires downloading embeddings, skip by default")
    def test_real_glove_embeddings(self):
        """Test with real GloVe embeddings (requires download)."""
        # This test is skipped by default to avoid downloading embeddings
        # Remove skipif to run with real embeddings
        encoder = EmbeddingEncoder(model_name='glove-wiki-gigaword-50')
        
        # Test basic encoding
        result = encoder.encode('hello world')
        
        assert result.shape == (50,)
        assert result.dtype == np.float32
        
        # Test that similar words have similar embeddings
        vec_king = encoder.encode('king')
        vec_queen = encoder.encode('queen')
        vec_apple = encoder.encode('apple')
        
        # Cosine similarity between king and queen should be higher than king and apple
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_king_queen = cosine_similarity(vec_king, vec_queen)
        sim_king_apple = cosine_similarity(vec_king, vec_apple)
        
        assert sim_king_queen > sim_king_apple


class TestEmbeddingEncoderRequirements:
    """Tests specifically validating requirements."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 3
            mock_model.__len__ = Mock(return_value=3)
            
            embeddings = {
                'hello': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'world': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_requirement_4_1_loads_pretrained_embeddings(self):
        """Requirement 4.1: Load pre-trained Word2Vec or GloVe embeddings."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 100
            mock_model.__len__ = Mock(return_value=400000)
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder(model_name='glove-wiki-gigaword-100')
            
            # Verify that gensim api.load was called with correct model name
            mock_load.assert_called_once_with('glove-wiki-gigaword-100')
            
            # Verify model is loaded
            assert encoder.model is not None
    
    def test_requirement_4_2_calculates_mean_vector(self, mock_encoder):
        """Requirement 4.2: Calculate mean vector of embeddings for all words."""
        # 'hello' = [1, 0, 0], 'world' = [0, 1, 0]
        # Mean should be [0.5, 0.5, 0]
        result = mock_encoder.encode('hello world')
        
        expected_mean = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected_mean)
    
    def test_requirement_4_3_handles_oov_with_zero_vector(self, mock_encoder):
        """Requirement 4.3: Handle OOV words by ignoring them or using zero vector."""
        # Test 1: All OOV words -> zero vector
        result_all_oov = mock_encoder.encode('unknown oov words')
        expected_zero = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result_all_oov, expected_zero)
        
        # Test 2: Mixed known and OOV -> OOV words ignored in mean
        # 'hello' = [1, 0, 0], 'unknown' is OOV, 'world' = [0, 1, 0]
        # Mean should only include 'hello' and 'world' -> [0.5, 0.5, 0]
        result_mixed = mock_encoder.encode('hello unknown world')
        expected_mixed = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result_mixed, expected_mixed)


class TestEmbeddingEncoderEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder."""
        with patch('src.embedding_encoder.api.load') as mock_load:
            mock_model = Mock()
            mock_model.vector_size = 3
            mock_model.__len__ = Mock(return_value=1)
            
            embeddings = {
                'hello': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            }
            
            mock_model.__contains__ = lambda self, word: word in embeddings
            mock_model.__getitem__ = lambda self, word: embeddings[word]
            
            mock_load.return_value = mock_model
            
            encoder = EmbeddingEncoder()
            return encoder
    
    def test_encode_very_long_text(self, mock_encoder):
        """Test encoding very long text."""
        # Create a long text with repeated words
        long_text = ' '.join(['hello'] * 1000)
        result = mock_encoder.encode(long_text)
        
        # Should still return correct dimension
        assert result.shape == (3,)
        
        # Should be same as single 'hello' since all words are the same
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_special_characters(self, mock_encoder):
        """Test encoding text with special characters."""
        # Special characters attached to words make them different tokens (OOV)
        # 'hello!!!' and 'world???' are different from 'hello' and 'world'
        result = mock_encoder.encode('hello!!! @#$ world???')
        
        # All tokens are OOV (hello!!! != hello, world??? != world)
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_numeric_strings(self, mock_encoder):
        """Test encoding numeric strings."""
        result = mock_encoder.encode('123 456 789')
        
        # Numbers are OOV
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_mixed_case(self, mock_encoder):
        """Test that encoding is case-sensitive."""
        # 'hello' is in vocab, 'Hello' and 'HELLO' are not
        result_lower = mock_encoder.encode('hello')
        result_upper = mock_encoder.encode('HELLO')
        result_title = mock_encoder.encode('Hello')
        
        # Only lowercase should match
        expected_lower = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        expected_zero = np.zeros(3, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result_lower, expected_lower)
        np.testing.assert_array_equal(result_upper, expected_zero)
        np.testing.assert_array_equal(result_title, expected_zero)
    
    def test_encode_single_character(self, mock_encoder):
        """Test encoding single character."""
        result = mock_encoder.encode('a')
        
        # Single character likely OOV
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
    
    def test_encode_repeated_word(self, mock_encoder):
        """Test encoding repeated word."""
        # 'hello hello hello' should give same result as 'hello'
        result_single = mock_encoder.encode('hello')
        result_repeated = mock_encoder.encode('hello hello hello')
        
        # Mean of identical vectors is the same vector
        np.testing.assert_array_almost_equal(result_single, result_repeated)
