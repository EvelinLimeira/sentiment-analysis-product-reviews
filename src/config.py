"""
Experiment configuration module for sentiment analysis NLP project.

This module defines the ExperimentConfig dataclass that holds all configuration
parameters for the sentiment analysis experiments, including settings for:
- Data splitting
- SVM + Bag of Words model
- SVM + Embeddings model
- BERT classifier
- In-Context Learning classifier
- Simulation parameters

"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import json


@dataclass
class ExperimentConfig:
    """
    Experiment configuration for sentiment analysis pipeline.
    
    This dataclass holds all hyperparameters and settings for running
    sentiment analysis experiments across multiple models.
    
    Attributes:
        dataset_name: Name of the dataset to use (e.g., 'amazon_reviews')
        train_size: Proportion of data for training (default 70%)
        val_size: Proportion of data for validation (default 15%)
        test_size: Proportion of data for testing (default 15%)
        num_simulations: Number of simulation runs for statistical validation (default 10)
        
        # SVM + BoW parameters
        tfidf_max_features: Maximum number of TF-IDF features
        tfidf_ngram_range: N-gram range for TF-IDF (unigrams and bigrams)
        svm_bow_kernel: SVM kernel type for BoW model
        svm_bow_C: Regularization parameter for BoW SVM
        
        # SVM + Embeddings parameters
        embedding_model: Pre-trained embedding model name (gensim)
        svm_emb_kernel: SVM kernel type for embeddings model
        svm_emb_C: Regularization parameter for embeddings SVM
        svm_emb_gamma: Kernel coefficient for RBF kernel
        
        # BERT parameters
        bert_model: Pre-trained BERT model name
        bert_max_length: Maximum sequence length for BERT tokenization
        bert_batch_size: Batch size for BERT training
        bert_epochs: Number of training epochs for BERT
        bert_learning_rate: Learning rate for BERT fine-tuning
        bert_patience: Early stopping patience (epochs without improvement)
        
        # ICL parameters
        icl_model: LLM model name for in-context learning
        icl_num_examples: Number of few-shot examples
        icl_sample_size: Sample size for ICL evaluation
    """
    
    # Dataset configuration
    dataset_name: str
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    num_simulations: int = 10
    
    # SVM + BoW configuration
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    svm_bow_kernel: str = 'linear'
    svm_bow_C: float = 1.0
    
    # SVM + Embeddings configuration
    embedding_model: str = 'glove-wiki-gigaword-100'
    svm_emb_kernel: str = 'rbf'
    svm_emb_C: float = 1.0
    svm_emb_gamma: str = 'scale'
    
    # BERT configuration
    bert_model: str = 'distilbert-base-uncased'
    bert_max_length: int = 512
    bert_batch_size: int = 32
    bert_epochs: int = 10
    bert_learning_rate: float = 2e-5
    bert_patience: int = 3  # Early stopping patience
    
    # ICL configuration
    icl_model: str = 'gpt-3.5-turbo'
    icl_num_examples: int = 5
    icl_sample_size: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_split_sizes()
        self._validate_positive_values()
    
    def _validate_split_sizes(self) -> None:
        """Validate that train/val/test sizes sum to 1.0."""
        total = self.train_size + self.val_size + self.test_size
        if not (0.99 <= total <= 1.01):  # Allow small floating point tolerance
            raise ValueError(
                f"Split sizes must sum to 1.0, got {total:.2f} "
                f"(train={self.train_size}, val={self.val_size}, test={self.test_size})"
            )
    
    def _validate_positive_values(self) -> None:
        """Validate that numeric parameters are positive."""
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.tfidf_max_features < 1:
            raise ValueError(f"tfidf_max_features must be >= 1, got {self.tfidf_max_features}")
        if self.svm_bow_C <= 0:
            raise ValueError(f"svm_bow_C must be > 0, got {self.svm_bow_C}")
        if self.svm_emb_C <= 0:
            raise ValueError(f"svm_emb_C must be > 0, got {self.svm_emb_C}")
        if self.bert_max_length < 1:
            raise ValueError(f"bert_max_length must be >= 1, got {self.bert_max_length}")
        if self.bert_batch_size < 1:
            raise ValueError(f"bert_batch_size must be >= 1, got {self.bert_batch_size}")
        if self.bert_epochs < 1:
            raise ValueError(f"bert_epochs must be >= 1, got {self.bert_epochs}")
        if self.bert_learning_rate <= 0:
            raise ValueError(f"bert_learning_rate must be > 0, got {self.bert_learning_rate}")
        if self.icl_num_examples < 1:
            raise ValueError(f"icl_num_examples must be >= 1, got {self.icl_num_examples}")
        if self.icl_sample_size < 1:
            raise ValueError(f"icl_sample_size must be >= 1, got {self.icl_sample_size}")
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            'dataset_name': self.dataset_name,
            'train_size': self.train_size,
            'val_size': self.val_size,
            'test_size': self.test_size,
            'num_simulations': self.num_simulations,
            'tfidf_max_features': self.tfidf_max_features,
            'tfidf_ngram_range': list(self.tfidf_ngram_range),
            'svm_bow_kernel': self.svm_bow_kernel,
            'svm_bow_C': self.svm_bow_C,
            'embedding_model': self.embedding_model,
            'svm_emb_kernel': self.svm_emb_kernel,
            'svm_emb_C': self.svm_emb_C,
            'svm_emb_gamma': self.svm_emb_gamma,
            'bert_model': self.bert_model,
            'bert_max_length': self.bert_max_length,
            'bert_batch_size': self.bert_batch_size,
            'bert_epochs': self.bert_epochs,
            'bert_learning_rate': self.bert_learning_rate,
            'icl_model': self.icl_model,
            'icl_num_examples': self.icl_num_examples,
            'icl_sample_size': self.icl_sample_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
            
        Returns:
            ExperimentConfig instance.
        """
        # Convert ngram_range list back to tuple if needed
        if 'tfidf_ngram_range' in config_dict:
            config_dict = config_dict.copy()
            config_dict['tfidf_ngram_range'] = tuple(config_dict['tfidf_ngram_range'])
        return cls(**config_dict)
    
    def to_json(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save the JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to the JSON file.
            
        Returns:
            ExperimentConfig instance.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_svm_bow_params(self) -> dict:
        """
        Get SVM + BoW specific parameters.
        
        Returns:
            Dictionary with SVM + BoW parameters.
        """
        return {
            'max_features': self.tfidf_max_features,
            'ngram_range': self.tfidf_ngram_range,
            'kernel': self.svm_bow_kernel,
            'C': self.svm_bow_C,
        }
    
    def get_svm_emb_params(self) -> dict:
        """
        Get SVM + Embeddings specific parameters.
        
        Returns:
            Dictionary with SVM + Embeddings parameters.
        """
        return {
            'embedding_model': self.embedding_model,
            'kernel': self.svm_emb_kernel,
            'C': self.svm_emb_C,
            'gamma': self.svm_emb_gamma,
        }
    
    def get_bert_params(self) -> dict:
        """
        Get BERT specific parameters.
        
        Returns:
            Dictionary with BERT parameters.
        """
        return {
            'model_name': self.bert_model,
            'max_length': self.bert_max_length,
            'batch_size': self.bert_batch_size,
            'epochs': self.bert_epochs,
            'learning_rate': self.bert_learning_rate,
        }
    
    def get_icl_params(self) -> dict:
        """
        Get ICL specific parameters.
        
        Returns:
            Dictionary with ICL parameters.
        """
        return {
            'model': self.icl_model,
            'num_examples': self.icl_num_examples,
            'sample_size': self.icl_sample_size,
        }
    
    def __str__(self) -> str:
        """Return string representation of configuration."""
        return (
            f"ExperimentConfig(\n"
            f"  dataset_name='{self.dataset_name}',\n"
            f"  splits=(train={self.train_size}, val={self.val_size}, test={self.test_size}),\n"
            f"  num_simulations={self.num_simulations},\n"
            f"  svm_bow=(max_features={self.tfidf_max_features}, ngram={self.tfidf_ngram_range}, "
            f"kernel='{self.svm_bow_kernel}', C={self.svm_bow_C}),\n"
            f"  svm_emb=(model='{self.embedding_model}', kernel='{self.svm_emb_kernel}', "
            f"C={self.svm_emb_C}, gamma='{self.svm_emb_gamma}'),\n"
            f"  bert=(model='{self.bert_model}', max_len={self.bert_max_length}, "
            f"batch={self.bert_batch_size}, epochs={self.bert_epochs}, lr={self.bert_learning_rate}),\n"
            f"  icl=(model='{self.icl_model}', examples={self.icl_num_examples}, "
            f"sample={self.icl_sample_size})\n"
            f")"
        )


def get_default_config(dataset_name: str = 'amazon_reviews') -> ExperimentConfig:
    """
    Get default experiment configuration.
    
    Args:
        dataset_name: Name of the dataset to use.
        
    Returns:
        ExperimentConfig with default parameters.
    """
    return ExperimentConfig(dataset_name=dataset_name)


def get_quick_test_config(dataset_name: str = 'amazon_reviews') -> ExperimentConfig:
    """
    Get configuration for quick testing with reduced parameters.
    
    Useful for debugging and quick validation runs.
    
    Args:
        dataset_name: Name of the dataset to use.
        
    Returns:
        ExperimentConfig with reduced parameters for faster execution.
    """
    return ExperimentConfig(
        dataset_name=dataset_name,
        num_simulations=2,
        tfidf_max_features=1000,
        bert_epochs=1,
        bert_batch_size=8,
        icl_sample_size=20,
    )
