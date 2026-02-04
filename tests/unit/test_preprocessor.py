"""
Unit tests for the Preprocessor class.

Tests cover:
- Lowercase conversion (Requirement 2.2)
- Special character removal (Requirement 2.3)
- Stopword removal (Requirement 2.4)
- Tokenization (Requirement 2.5)
- Fit/transform pattern (Requirements 2.1, 2.6)
- OOV handling (Requirement 2.7)
"""

import pytest
from src.preprocessor import Preprocessor


class TestPreprocessorInit:
    """Tests for Preprocessor initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        preprocessor = Preprocessor()
        assert preprocessor.language == 'english'
        assert preprocessor.remove_stopwords is True
        assert preprocessor.is_fitted is False
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        preprocessor = Preprocessor(language='english', remove_stopwords=False)
        assert preprocessor.language == 'english'
        assert preprocessor.remove_stopwords is False
    
    def test_stopwords_loaded_for_english(self):
        """English stopwords should be loaded by default."""
        preprocessor = Preprocessor(language='english')
        stopwords = preprocessor.get_stopwords()
        
        # Check some common stopwords are present
        assert 'the' in stopwords
        assert 'is' in stopwords
        assert 'and' in stopwords
        assert 'a' in stopwords


class TestLowercaseConversion:
    """Tests for lowercase conversion (Requirement 2.2)."""
    
    @pytest.fixture
    def fitted_preprocessor(self):
        """Create a fitted preprocessor."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['sample text'])
        return preprocessor
    
    def test_uppercase_converted_to_lowercase(self, fitted_preprocessor):
        """All uppercase letters should be converted to lowercase."""
        result = fitted_preprocessor.transform(['HELLO WORLD'])
        assert result[0] == 'hello world'
    
    def test_mixed_case_converted_to_lowercase(self, fitted_preprocessor):
        """Mixed case should be converted to lowercase."""
        result = fitted_preprocessor.transform(['HeLLo WoRLd'])
        assert result[0] == 'hello world'
    
    def test_already_lowercase_unchanged(self, fitted_preprocessor):
        """Already lowercase text should remain unchanged."""
        result = fitted_preprocessor.transform(['hello world'])
        assert result[0] == 'hello world'
    
    def test_no_uppercase_in_output(self, fitted_preprocessor):
        """Output should contain no uppercase ASCII characters."""
        result = fitted_preprocessor.transform(['THIS IS A TEST With MIXED Case'])
        # Check no uppercase letters A-Z in output
        assert result[0] == result[0].lower()
        assert not any(c.isupper() for c in result[0])


class TestSpecialCharacterRemoval:
    """Tests for special character removal (Requirement 2.3)."""
    
    @pytest.fixture
    def fitted_preprocessor(self):
        """Create a fitted preprocessor without stopword removal."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['sample text'])
        return preprocessor
    
    def test_punctuation_removed(self, fitted_preprocessor):
        """Punctuation should be removed."""
        result = fitted_preprocessor.transform(['hello, world! how are you?'])
        assert ',' not in result[0]
        assert '!' not in result[0]
        assert '?' not in result[0]
    
    def test_special_characters_removed(self, fitted_preprocessor):
        """Special characters should be removed."""
        result = fitted_preprocessor.transform(['hello@world #test $money %percent'])
        assert '@' not in result[0]
        assert '#' not in result[0]
        assert '$' not in result[0]
        assert '%' not in result[0]
    
    def test_numbers_preserved(self, fitted_preprocessor):
        """Numbers should be preserved."""
        result = fitted_preprocessor.transform(['product 123 review'])
        assert '123' in result[0]
    
    def test_excessive_whitespace_normalized(self, fitted_preprocessor):
        """Excessive whitespace should be normalized to single spaces."""
        result = fitted_preprocessor.transform(['hello    world   test'])
        assert '  ' not in result[0]  # No double spaces
        assert 'hello world test' == result[0]
    
    def test_leading_trailing_whitespace_removed(self, fitted_preprocessor):
        """Leading and trailing whitespace should be removed."""
        result = fitted_preprocessor.transform(['  hello world  '])
        assert result[0] == 'hello world'


class TestStopwordRemoval:
    """Tests for stopword removal (Requirement 2.4)."""
    
    @pytest.fixture
    def fitted_preprocessor_with_stopwords(self):
        """Create a fitted preprocessor with stopword removal enabled."""
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(['sample text for training'])
        return preprocessor
    
    @pytest.fixture
    def fitted_preprocessor_without_stopwords(self):
        """Create a fitted preprocessor with stopword removal disabled."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['sample text for training'])
        return preprocessor
    
    def test_stopwords_removed_when_enabled(self, fitted_preprocessor_with_stopwords):
        """Stopwords should be removed when remove_stopwords=True."""
        result = fitted_preprocessor_with_stopwords.transform(['the quick brown fox'])
        tokens = result[0].split()
        
        # 'the' is a stopword and should be removed
        assert 'the' not in tokens
        # 'quick', 'brown', 'fox' are not stopwords
        assert 'quick' in tokens
        assert 'brown' in tokens
        assert 'fox' in tokens
    
    def test_stopwords_preserved_when_disabled(self, fitted_preprocessor_without_stopwords):
        """Stopwords should be preserved when remove_stopwords=False."""
        result = fitted_preprocessor_without_stopwords.transform(['the quick brown fox'])
        tokens = result[0].split()
        
        # 'the' should be preserved
        assert 'the' in tokens
    
    def test_common_stopwords_removed(self, fitted_preprocessor_with_stopwords):
        """Common stopwords should be removed."""
        result = fitted_preprocessor_with_stopwords.transform(
            ['this is a test and it should work']
        )
        tokens = result[0].split()
        
        # These are all stopwords
        stopwords_to_check = ['this', 'is', 'a', 'and', 'it', 'should']
        for stopword in stopwords_to_check:
            assert stopword not in tokens
        
        # 'test' and 'work' are not stopwords
        assert 'test' in tokens
        assert 'work' in tokens
    
    def test_output_contains_no_stopwords(self, fitted_preprocessor_with_stopwords):
        """Output tokens should not contain any stopwords from the predefined list."""
        preprocessor = fitted_preprocessor_with_stopwords
        stopwords = preprocessor.get_stopwords()
        
        result = preprocessor.transform(['the quick brown fox jumps over the lazy dog'])
        tokens = result[0].split()
        
        for token in tokens:
            assert token not in stopwords, f"Stopword '{token}' found in output"


class TestTokenization:
    """Tests for tokenization (Requirement 2.5)."""
    
    @pytest.fixture
    def fitted_preprocessor(self):
        """Create a fitted preprocessor."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['sample text'])
        return preprocessor
    
    def test_text_tokenized_by_whitespace(self, fitted_preprocessor):
        """Text should be tokenized by whitespace."""
        result = fitted_preprocessor.transform(['hello world test'])
        tokens = result[0].split()
        
        assert len(tokens) == 3
        assert tokens == ['hello', 'world', 'test']
    
    def test_empty_text_returns_empty_string(self, fitted_preprocessor):
        """Empty text should return empty string."""
        result = fitted_preprocessor.transform([''])
        assert result[0] == ''
    
    def test_whitespace_only_returns_empty_string(self, fitted_preprocessor):
        """Whitespace-only text should return empty string."""
        result = fitted_preprocessor.transform(['   '])
        assert result[0] == ''


class TestFitTransformPattern:
    """Tests for fit/transform pattern (Requirements 2.1, 2.6)."""
    
    def test_transform_before_fit_raises_error(self):
        """Calling transform before fit should raise RuntimeError."""
        preprocessor = Preprocessor()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            preprocessor.transform(['test text'])
    
    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        preprocessor = Preprocessor()
        result = preprocessor.fit(['sample text'])
        
        assert result is preprocessor
    
    def test_fit_sets_is_fitted_flag(self):
        """fit() should set is_fitted to True."""
        preprocessor = Preprocessor()
        assert preprocessor.is_fitted is False
        
        preprocessor.fit(['sample text'])
        assert preprocessor.is_fitted is True
    
    def test_fit_builds_vocabulary(self):
        """fit() should build vocabulary from training texts."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['hello world', 'world test'])
        
        assert 'hello' in preprocessor.vocabulary
        assert 'world' in preprocessor.vocabulary
        assert 'test' in preprocessor.vocabulary
    
    def test_fit_transform_convenience_method(self):
        """fit_transform() should fit and transform in one step."""
        preprocessor = Preprocessor(remove_stopwords=False)
        result = preprocessor.fit_transform(['HELLO WORLD'])
        
        assert preprocessor.is_fitted is True
        assert result[0] == 'hello world'
    
    def test_transform_uses_same_transformations(self):
        """transform() should apply same transformations to all texts."""
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(['training text sample'])
        
        # Transform should apply same preprocessing
        result1 = preprocessor.transform(['THE QUICK BROWN FOX'])
        result2 = preprocessor.transform(['THE QUICK BROWN FOX'])
        
        assert result1 == result2


class TestOOVHandling:
    """Tests for OOV (out-of-vocabulary) handling (Requirement 2.7)."""
    
    def test_oov_words_kept_in_output(self):
        """OOV words should be kept in output (not filtered)."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['hello world'])  # Only 'hello' and 'world' in vocab
        
        # 'unknown' is OOV but should still appear in output
        result = preprocessor.transform(['hello unknown word'])
        tokens = result[0].split()
        
        assert 'unknown' in tokens
        assert 'word' in tokens
    
    def test_is_oov_method(self):
        """is_oov() should correctly identify OOV words."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['hello world'])
        
        assert preprocessor.is_oov('hello') is False
        assert preprocessor.is_oov('world') is False
        assert preprocessor.is_oov('unknown') is True
    
    def test_is_oov_before_fit_raises_error(self):
        """is_oov() before fit should raise RuntimeError."""
        preprocessor = Preprocessor()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            preprocessor.is_oov('test')
    
    def test_vocabulary_size_method(self):
        """get_vocabulary_size() should return correct count."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['hello world', 'world test'])
        
        # Vocabulary: hello, world, test
        assert preprocessor.get_vocabulary_size() == 3
    
    def test_vocabulary_size_before_fit_raises_error(self):
        """get_vocabulary_size() before fit should raise RuntimeError."""
        preprocessor = Preprocessor()
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            preprocessor.get_vocabulary_size()


class TestStopwordManagement:
    """Tests for stopword management methods."""
    
    def test_add_stopwords(self):
        """add_stopwords() should add words to stopword set."""
        preprocessor = Preprocessor()
        original_count = len(preprocessor.get_stopwords())
        
        preprocessor.add_stopwords(['customword', 'anotherword'])
        
        assert len(preprocessor.get_stopwords()) == original_count + 2
        assert 'customword' in preprocessor.get_stopwords()
        assert 'anotherword' in preprocessor.get_stopwords()
    
    def test_remove_from_stopwords(self):
        """remove_from_stopwords() should remove words from stopword set."""
        preprocessor = Preprocessor()
        
        # 'the' is a default stopword
        assert 'the' in preprocessor.get_stopwords()
        
        preprocessor.remove_from_stopwords(['the'])
        
        assert 'the' not in preprocessor.get_stopwords()
    
    def test_get_stopwords_returns_copy(self):
        """get_stopwords() should return a copy, not the original set."""
        preprocessor = Preprocessor()
        stopwords = preprocessor.get_stopwords()
        
        # Modifying the returned set should not affect the original
        stopwords.add('newword')
        
        assert 'newword' not in preprocessor.get_stopwords()


class TestEdgeCases:
    """Tests for edge cases and special inputs."""
    
    @pytest.fixture
    def fitted_preprocessor(self):
        """Create a fitted preprocessor."""
        preprocessor = Preprocessor(remove_stopwords=False)
        preprocessor.fit(['sample text'])
        return preprocessor
    
    def test_none_input_returns_empty_string(self, fitted_preprocessor):
        """None input should return empty string."""
        result = fitted_preprocessor.transform([None])
        assert result[0] == ''
    
    def test_numeric_input_converted_to_string(self, fitted_preprocessor):
        """Numeric input should be converted to string."""
        result = fitted_preprocessor.transform([12345])
        assert result[0] == '12345'
    
    def test_empty_list_returns_empty_list(self, fitted_preprocessor):
        """Empty input list should return empty list."""
        result = fitted_preprocessor.transform([])
        assert result == []
    
    def test_multiple_texts_processed(self, fitted_preprocessor):
        """Multiple texts should all be processed."""
        result = fitted_preprocessor.transform(['TEXT ONE', 'TEXT TWO', 'TEXT THREE'])
        
        assert len(result) == 3
        assert result[0] == 'text one'
        assert result[1] == 'text two'
        assert result[2] == 'text three'
    
    def test_unicode_text_handled(self, fitted_preprocessor):
        """Unicode text should be handled (special chars removed)."""
        result = fitted_preprocessor.transform(['café résumé naïve'])
        # Special characters like é, ï should be removed
        # Only alphanumeric and spaces remain
        assert 'caf' in result[0] or 'cafe' in result[0] or result[0] == 'caf rsum nave'
    
    def test_emoji_removed(self, fitted_preprocessor):
        """Emojis should be removed as special characters."""
        result = fitted_preprocessor.transform(['great product 😀👍'])
        assert '😀' not in result[0]
        assert '👍' not in result[0]
        assert 'great' in result[0]
        assert 'product' in result[0]


class TestIntegration:
    """Integration tests for complete preprocessing pipeline."""
    
    def test_complete_preprocessing_pipeline(self):
        """Test complete preprocessing with all steps."""
        preprocessor = Preprocessor(remove_stopwords=True)
        
        # Fit on training data
        training_texts = [
            "This is a GREAT product!",
            "The quality is amazing.",
            "I love this item!!!"
        ]
        preprocessor.fit(training_texts)
        
        # Transform test data
        test_texts = [
            "THIS IS THE BEST PRODUCT EVER!!!",
            "I really love it, amazing quality."
        ]
        result = preprocessor.transform(test_texts)
        
        # Check results
        assert len(result) == 2
        
        # All lowercase
        for text in result:
            assert text == text.lower()
        
        # No special characters
        for text in result:
            assert '!' not in text
            assert ',' not in text
            assert '.' not in text
        
        # Stopwords removed
        stopwords = preprocessor.get_stopwords()
        for text in result:
            tokens = text.split()
            for token in tokens:
                assert token not in stopwords
    
    def test_preprocessing_consistency_across_calls(self):
        """Preprocessing should be consistent across multiple calls."""
        preprocessor = Preprocessor(remove_stopwords=True)
        preprocessor.fit(['training text'])
        
        text = "The Quick Brown Fox Jumps Over The Lazy Dog!"
        
        result1 = preprocessor.transform([text])
        result2 = preprocessor.transform([text])
        result3 = preprocessor.transform([text])
        
        assert result1 == result2 == result3
