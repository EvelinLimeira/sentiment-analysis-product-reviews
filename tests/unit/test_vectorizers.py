"""
Unit tests for the BoWVectorizer class.

Tests cover:
- Initialization and parameter validation
- Fit on training only (Requirement 3.1)
- TF-IDF vectorization (Requirement 3.2)
- Vocabulary consistency for test data (Requirement 3.3)
"""

import pytest
import numpy as np
import scipy.sparse
from src.vectorizers import BoWVectorizer


class TestBoWVectorizerInit:
    """Tests for BoWVectorizer initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        vectorizer = BoWVectorizer()
        assert vectorizer.max_features == 5000
        assert vectorizer.ngram_range == (1, 2)
        assert vectorizer.is_fitted is False
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        vectorizer = BoWVectorizer(max_features=1000, ngram_range=(1, 3))
        assert vectorizer.max_features == 1000
        assert vectorizer.ngram_range == (1, 3)
    
    def test_invalid_max_features_raises_error(self):
        """max_features < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            BoWVectorizer(max_features=0)
        
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            BoWVectorizer(max_features=-1)
    
    def test_invalid_ngram_range_raises_error(self):
        """Invalid ngram_range should raise ValueError."""
        # Not a tuple
        with pytest.raises(ValueError, match="ngram_range must be a tuple"):
            BoWVectorizer(ngram_range=[1, 2])
        
        # Wrong length
        with pytest.raises(ValueError, match="ngram_range must be a tuple of 2"):
            BoWVectorizer(ngram_range=(1,))
        
        # Min < 1
        with pytest.raises(ValueError, match="ngram_range must have min >= 1"):
            BoWVectorizer(ngram_range=(0, 2))
        
        # Max < min
        with pytest.raises(ValueError, match="ngram_range must have min >= 1 and max >= min"):
            BoWVectorizer(ngram_range=(3, 2))


class TestBoWVectorizerFit:
    """Tests for fit() method (Requirement 3.1)."""
    
    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        vectorizer = BoWVectorizer()
        result = vectorizer.fit(['sample text', 'another text'])
        assert result is vectorizer
    
    def test_fit_sets_is_fitted_flag(self):
        """fit() should set is_fitted to True."""
        vectorizer = BoWVectorizer()
        assert vectorizer.is_fitted is False
        
        vectorizer.fit(['sample text'])
        assert vectorizer.is_fitted is True
    
    def test_fit_builds_vocabulary(self):
        """fit() should build vocabulary from training texts."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(['hello world', 'world test'])
        
        vocab = vectorizer.get_vocabulary()
        assert 'hello' in vocab
        assert 'world' in vocab
        assert 'test' in vocab
    
    def test_fit_on_empty_list_raises_error(self):
        """fit() on empty list should raise ValueError."""
        vectorizer = BoWVectorizer()
        
        with pytest.raises(ValueError, match="Cannot fit on empty"):
            vectorizer.fit([])
    
    def test_fit_respects_max_features(self):
        """fit() should respect max_features limit."""
        # Create texts with many unique words
        texts = [f"word{i}" for i in range(100)]
        
        vectorizer = BoWVectorizer(max_features=10, ngram_range=(1, 1))
        vectorizer.fit(texts)
        
        assert vectorizer.get_vocabulary_size() <= 10
    
    def test_fit_creates_ngrams(self):
        """fit() should create n-grams according to ngram_range."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 2))
        vectorizer.fit(['hello world test'])
        
        vocab = vectorizer.get_vocabulary()
        
        # Should have unigrams
        assert 'hello' in vocab
        assert 'world' in vocab
        assert 'test' in vocab
        
        # Should have bigrams
        assert 'hello world' in vocab
        assert 'world test' in vocab


class TestBoWVectorizerTransform:
    """Tests for transform() method (Requirements 3.2, 3.3)."""
    
    @pytest.fixture
    def fitted_vectorizer(self):
        """Create a fitted vectorizer."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 2))
        vectorizer.fit(['hello world', 'world test', 'hello test'])
        return vectorizer
    
    def test_transform_before_fit_raises_error(self):
        """transform() before fit should raise RuntimeError."""
        vectorizer = BoWVectorizer()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            vectorizer.transform(['test text'])
    
    def test_transform_returns_sparse_matrix(self, fitted_vectorizer):
        """transform() should return a sparse CSR matrix."""
        result = fitted_vectorizer.transform(['hello world'])
        
        assert isinstance(result, scipy.sparse.csr_matrix)
    
    def test_transform_output_shape(self, fitted_vectorizer):
        """transform() output should have correct shape."""
        texts = ['hello world', 'test text', 'another text']
        result = fitted_vectorizer.transform(texts)
        
        # Shape should be (n_texts, n_features)
        assert result.shape[0] == 3
        assert result.shape[1] == fitted_vectorizer.get_vocabulary_size()
    
    def test_transform_values_non_negative(self, fitted_vectorizer):
        """TF-IDF values should be non-negative."""
        result = fitted_vectorizer.transform(['hello world test'])
        
        # All values should be >= 0
        assert (result.toarray() >= 0).all()
    
    def test_transform_values_are_floats(self, fitted_vectorizer):
        """TF-IDF values should be floats."""
        result = fitted_vectorizer.transform(['hello world'])
        
        assert result.dtype in [np.float32, np.float64]
    
    def test_transform_empty_text_returns_zero_vector(self, fitted_vectorizer):
        """Empty text should return zero vector."""
        result = fitted_vectorizer.transform([''])
        
        # All values should be 0
        assert result.sum() == 0
    
    def test_transform_oov_words_ignored(self, fitted_vectorizer):
        """Words not in vocabulary should be ignored (zero weight)."""
        # 'unknown' and 'words' are not in the training vocabulary
        result = fitted_vectorizer.transform(['unknown words here'])
        
        # The result should be mostly zeros (only 'here' might match if in vocab)
        # Since 'unknown', 'words', 'here' are likely not in vocab, sum should be 0 or very small
        vocab = fitted_vectorizer.get_vocabulary()
        
        # Check that unknown words don't create new features
        assert result.shape[1] == fitted_vectorizer.get_vocabulary_size()
    
    def test_transform_uses_training_vocabulary_only(self):
        """transform() should use ONLY vocabulary from training (Requirement 3.3)."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        
        # Fit on limited vocabulary
        vectorizer.fit(['apple banana'])
        vocab_after_fit = vectorizer.get_vocabulary().copy()
        
        # Transform with new words
        vectorizer.transform(['apple cherry date'])
        vocab_after_transform = vectorizer.get_vocabulary()
        
        # Vocabulary should not change after transform
        assert vocab_after_fit == vocab_after_transform
        
        # New words should not be in vocabulary
        assert 'cherry' not in vocab_after_transform
        assert 'date' not in vocab_after_transform


class TestBoWVectorizerFitTransform:
    """Tests for fit_transform() method."""
    
    def test_fit_transform_equivalent_to_fit_then_transform(self):
        """fit_transform() should be equivalent to fit() then transform()."""
        texts = ['hello world', 'world test', 'hello test']
        
        # Method 1: fit_transform
        vectorizer1 = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        result1 = vectorizer1.fit_transform(texts)
        
        # Method 2: fit then transform
        vectorizer2 = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer2.fit(texts)
        result2 = vectorizer2.transform(texts)
        
        # Results should be equal
        np.testing.assert_array_almost_equal(
            result1.toarray(), result2.toarray()
        )
    
    def test_fit_transform_sets_is_fitted(self):
        """fit_transform() should set is_fitted to True."""
        vectorizer = BoWVectorizer()
        vectorizer.fit_transform(['sample text'])
        
        assert vectorizer.is_fitted is True


class TestBoWVectorizerVocabularyMethods:
    """Tests for vocabulary-related methods."""
    
    @pytest.fixture
    def fitted_vectorizer(self):
        """Create a fitted vectorizer."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(['hello world', 'world test'])
        return vectorizer
    
    def test_get_vocabulary_before_fit_raises_error(self):
        """get_vocabulary() before fit should raise RuntimeError."""
        vectorizer = BoWVectorizer()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            vectorizer.get_vocabulary()
    
    def test_get_vocabulary_returns_copy(self, fitted_vectorizer):
        """get_vocabulary() should return a copy."""
        vocab1 = fitted_vectorizer.get_vocabulary()
        vocab1['newword'] = 999
        
        vocab2 = fitted_vectorizer.get_vocabulary()
        assert 'newword' not in vocab2
    
    def test_get_vocabulary_size(self, fitted_vectorizer):
        """get_vocabulary_size() should return correct count."""
        vocab = fitted_vectorizer.get_vocabulary()
        size = fitted_vectorizer.get_vocabulary_size()
        
        assert size == len(vocab)
    
    def test_get_vocabulary_size_before_fit_raises_error(self):
        """get_vocabulary_size() before fit should raise RuntimeError."""
        vectorizer = BoWVectorizer()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            vectorizer.get_vocabulary_size()
    
    def test_get_feature_names(self, fitted_vectorizer):
        """get_feature_names() should return list of feature names."""
        feature_names = fitted_vectorizer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) == fitted_vectorizer.get_vocabulary_size()
        assert 'hello' in feature_names
        assert 'world' in feature_names
        assert 'test' in feature_names
    
    def test_get_feature_names_before_fit_raises_error(self):
        """get_feature_names() before fit should raise RuntimeError."""
        vectorizer = BoWVectorizer()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            vectorizer.get_feature_names()
    
    def test_is_in_vocabulary(self, fitted_vectorizer):
        """is_in_vocabulary() should correctly identify vocabulary terms."""
        assert fitted_vectorizer.is_in_vocabulary('hello') is True
        assert fitted_vectorizer.is_in_vocabulary('world') is True
        assert fitted_vectorizer.is_in_vocabulary('unknown') is False
    
    def test_is_in_vocabulary_before_fit_raises_error(self):
        """is_in_vocabulary() before fit should raise RuntimeError."""
        vectorizer = BoWVectorizer()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            vectorizer.is_in_vocabulary('test')


class TestBoWVectorizerIDF:
    """Tests for IDF-related functionality."""
    
    @pytest.fixture
    def fitted_vectorizer(self):
        """Create a fitted vectorizer with known IDF values."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        # 'common' appears in all docs, 'rare' in one
        vectorizer.fit([
            'common word',
            'common another',
            'common rare'
        ])
        return vectorizer
    
    def test_get_idf_returns_float(self, fitted_vectorizer):
        """get_idf() should return a float for known terms."""
        idf = fitted_vectorizer.get_idf('common')
        
        assert isinstance(idf, float)
        assert idf > 0
    
    def test_get_idf_returns_none_for_unknown(self, fitted_vectorizer):
        """get_idf() should return None for unknown terms."""
        idf = fitted_vectorizer.get_idf('unknown')
        
        assert idf is None
    
    def test_get_idf_before_fit_raises_error(self):
        """get_idf() before fit should raise RuntimeError."""
        vectorizer = BoWVectorizer()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            vectorizer.get_idf('test')
    
    def test_rare_words_have_higher_idf(self, fitted_vectorizer):
        """Rare words should have higher IDF than common words."""
        idf_common = fitted_vectorizer.get_idf('common')
        idf_rare = fitted_vectorizer.get_idf('rare')
        
        # 'rare' appears in fewer documents, so should have higher IDF
        assert idf_rare > idf_common


class TestBoWVectorizerTFIDFValues:
    """Tests for TF-IDF value correctness (Requirement 3.2)."""
    
    def test_tfidf_values_in_valid_range(self):
        """TF-IDF values should be non-negative floats."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(['hello world', 'world test', 'hello test'])
        
        result = vectorizer.transform(['hello world test'])
        values = result.toarray()
        
        # All values should be >= 0
        assert (values >= 0).all()
        
        # Values should be floats
        assert values.dtype in [np.float32, np.float64]
    
    def test_tfidf_same_dimension_as_vocabulary(self):
        """TF-IDF vectors should have same dimension as vocabulary size."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(['hello world', 'world test'])
        
        result = vectorizer.transform(['hello'])
        vocab_size = vectorizer.get_vocabulary_size()
        
        assert result.shape[1] == vocab_size
    
    def test_tfidf_consistent_across_transforms(self):
        """Same text should produce same TF-IDF vector."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(['hello world', 'world test'])
        
        text = ['hello world']
        result1 = vectorizer.transform(text)
        result2 = vectorizer.transform(text)
        
        np.testing.assert_array_equal(result1.toarray(), result2.toarray())


class TestBoWVectorizerNgrams:
    """Tests for n-gram functionality."""
    
    def test_unigrams_only(self):
        """ngram_range=(1,1) should create only unigrams."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(['hello world test'])
        
        vocab = vectorizer.get_vocabulary()
        
        # Should have unigrams
        assert 'hello' in vocab
        assert 'world' in vocab
        assert 'test' in vocab
        
        # Should NOT have bigrams
        assert 'hello world' not in vocab
        assert 'world test' not in vocab
    
    def test_bigrams_only(self):
        """ngram_range=(2,2) should create only bigrams."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(2, 2))
        vectorizer.fit(['hello world test'])
        
        vocab = vectorizer.get_vocabulary()
        
        # Should NOT have unigrams
        assert 'hello' not in vocab
        assert 'world' not in vocab
        assert 'test' not in vocab
        
        # Should have bigrams
        assert 'hello world' in vocab
        assert 'world test' in vocab
    
    def test_unigrams_and_bigrams(self):
        """ngram_range=(1,2) should create unigrams and bigrams."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 2))
        vectorizer.fit(['hello world test'])
        
        vocab = vectorizer.get_vocabulary()
        
        # Should have unigrams
        assert 'hello' in vocab
        assert 'world' in vocab
        assert 'test' in vocab
        
        # Should have bigrams
        assert 'hello world' in vocab
        assert 'world test' in vocab
    
    def test_trigrams(self):
        """ngram_range=(1,3) should include trigrams."""
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 3))
        vectorizer.fit(['hello world test'])
        
        vocab = vectorizer.get_vocabulary()
        
        # Should have trigram
        assert 'hello world test' in vocab


class TestBoWVectorizerIntegration:
    """Integration tests for complete vectorization pipeline."""
    
    def test_complete_vectorization_pipeline(self):
        """Test complete vectorization with realistic data."""
        # Training data
        train_texts = [
            "great product love it",
            "terrible quality waste money",
            "amazing service recommend",
            "poor customer support",
            "excellent value price"
        ]
        
        # Test data (includes OOV words)
        test_texts = [
            "great service excellent",
            "terrible support awful"
        ]
        
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 2))
        
        # Fit on training
        vectorizer.fit(train_texts)
        
        # Transform both sets
        train_vectors = vectorizer.transform(train_texts)
        test_vectors = vectorizer.transform(test_texts)
        
        # Check shapes
        assert train_vectors.shape[0] == len(train_texts)
        assert test_vectors.shape[0] == len(test_texts)
        assert train_vectors.shape[1] == test_vectors.shape[1]
        
        # Check values are valid
        assert (train_vectors.toarray() >= 0).all()
        assert (test_vectors.toarray() >= 0).all()
    
    def test_vocabulary_not_modified_by_test_data(self):
        """Vocabulary should not be modified when transforming test data."""
        train_texts = ['apple banana cherry']
        test_texts = ['apple date elderberry fig']  # date, elderberry, fig are OOV
        
        vectorizer = BoWVectorizer(max_features=100, ngram_range=(1, 1))
        vectorizer.fit(train_texts)
        
        vocab_before = vectorizer.get_vocabulary().copy()
        vectorizer.transform(test_texts)
        vocab_after = vectorizer.get_vocabulary()
        
        # Vocabulary should be unchanged
        assert vocab_before == vocab_after
        
        # OOV words should not be in vocabulary
        assert 'date' not in vocab_after
        assert 'elderberry' not in vocab_after
        assert 'fig' not in vocab_after
    
    def test_sparse_matrix_efficiency(self):
        """Sparse matrix should be memory efficient for sparse data."""
        # Create texts that will produce sparse vectors
        texts = [f"word{i}" for i in range(100)]
        
        vectorizer = BoWVectorizer(max_features=1000, ngram_range=(1, 1))
        vectorizer.fit(texts)
        
        result = vectorizer.transform(['word0 word1'])
        
        # Should be sparse (most values are 0)
        dense = result.toarray()
        non_zero_count = np.count_nonzero(dense)
        total_count = dense.size
        
        # Less than 10% should be non-zero
        assert non_zero_count / total_count < 0.1
