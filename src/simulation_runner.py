"""
Simulation runner module for sentiment analysis NLP project.

This module provides functionality to run multiple simulations (N runs with different
random seeds) for statistical validation of model performance. Each simulation trains
and evaluates models with different data splits and initializations.

Requirements: 7.1, 7.2, 7.8
"""

import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.vectorizers import BoWVectorizer
from src.embedding_encoder import EmbeddingEncoder
from src.classifiers import SVMClassifier
from src.bert_classifier import BERTClassifier
from src.evaluator import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """
    Result from a single simulation run.
    
    Attributes:
        simulation_id: Unique identifier for this simulation (0-indexed)
        model_name: Name of the model (e.g., 'svm_bow', 'svm_embeddings', 'bert')
        random_seed: Random seed used for this simulation
        accuracy: Accuracy score on test set
        precision_macro: Macro-averaged precision
        recall_macro: Macro-averaged recall
        f1_macro: Macro-averaged F1 score
        f1_weighted: Weighted F1 score
        training_time: Training time in seconds
        inference_time: Inference time in seconds
    """
    simulation_id: int
    model_name: str
    random_seed: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    training_time: float
    inference_time: float


class SimulationRunner:
    """
    Runs multiple simulations for statistical validation.
    
    This class orchestrates running N simulations per model with different
    random seeds for data splitting and model initialization. Results are
    stored in CSV format for statistical analysis.
    """
    
    def __init__(self, config: ExperimentConfig, output_dir: str = 'results/simulations'):
        """
        Initialize the simulation runner.
        
        Args:
            config: Experiment configuration
            output_dir: Directory to save simulation results (default: 'results/simulations')
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SimulationRunner initialized with {config.num_simulations} simulations")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def run_single_simulation(
        self,
        model_name: str,
        simulation_id: int,
        random_seed: int
    ) -> SimulationResult:
        """
        Run a single simulation for a specific model.
        
        Args:
            model_name: Name of the model ('svm_bow', 'svm_embeddings', or 'bert')
            simulation_id: Simulation identifier (0-indexed)
            random_seed: Random seed for this simulation
            
        Returns:
            SimulationResult with all metrics
            
        Raises:
            ValueError: If model_name is not recognized
        """
        logger.info(f"Running simulation {simulation_id} for {model_name} (seed={random_seed})")
        
        # Load data with the specified random seed
        data_loader = DataLoader(
            dataset_name=self.config.dataset_name,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=random_seed
        )
        train_df, val_df, test_df = data_loader.load()
        
        # Extract texts and labels
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].values
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].values
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].values
        
        # Train and evaluate based on model type
        if model_name == 'svm_bow':
            result = self._run_svm_bow(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                simulation_id, random_seed
            )
        elif model_name == 'svm_embeddings':
            result = self._run_svm_embeddings(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                simulation_id, random_seed
            )
        elif model_name == 'bert':
            result = self._run_bert(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                simulation_id, random_seed
            )
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Expected 'svm_bow', 'svm_embeddings', or 'bert'"
            )
        
        logger.info(
            f"Simulation {simulation_id} complete: "
            f"accuracy={result.accuracy:.4f}, f1_macro={result.f1_macro:.4f}"
        )
        
        return result
    
    def _run_svm_bow(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        test_texts: List[str],
        test_labels: np.ndarray,
        simulation_id: int,
        random_seed: int
    ) -> SimulationResult:
        """Run SVM with Bag of Words model."""
        # Preprocess texts
        preprocessor = Preprocessor(language='english', remove_stopwords=True)
        train_texts_processed = preprocessor.fit_transform(train_texts)
        test_texts_processed = preprocessor.transform(test_texts)
        
        # Vectorize with TF-IDF
        vectorizer = BoWVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range
        )
        
        # Training
        start_time = time.time()
        X_train = vectorizer.fit_transform(train_texts_processed)
        classifier = SVMClassifier(
            kernel=self.config.svm_bow_kernel,
            C=self.config.svm_bow_C
        )
        classifier.fit(X_train, train_labels)
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        X_test = vectorizer.transform(test_texts_processed)
        predictions = classifier.predict(X_test)
        inference_time = time.time() - start_time
        
        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.evaluate(test_labels, predictions, 'svm_bow')
        
        return SimulationResult(
            simulation_id=simulation_id,
            model_name='svm_bow',
            random_seed=random_seed,
            accuracy=metrics['accuracy'],
            precision_macro=metrics['precision_macro'],
            recall_macro=metrics['recall_macro'],
            f1_macro=metrics['f1_macro'],
            f1_weighted=metrics['f1_weighted'],
            training_time=training_time,
            inference_time=inference_time
        )
    
    def _run_svm_embeddings(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        test_texts: List[str],
        test_labels: np.ndarray,
        simulation_id: int,
        random_seed: int
    ) -> SimulationResult:
        """Run SVM with Embeddings model."""
        # Preprocess texts
        preprocessor = Preprocessor(language='english', remove_stopwords=True)
        train_texts_processed = preprocessor.fit_transform(train_texts)
        test_texts_processed = preprocessor.transform(test_texts)
        
        # Encode with embeddings
        encoder = EmbeddingEncoder(model_name=self.config.embedding_model)
        
        # Training
        start_time = time.time()
        X_train = encoder.encode_batch(train_texts_processed)
        classifier = SVMClassifier(
            kernel=self.config.svm_emb_kernel,
            C=self.config.svm_emb_C,
            gamma=self.config.svm_emb_gamma
        )
        classifier.fit(X_train, train_labels)
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        X_test = encoder.encode_batch(test_texts_processed)
        predictions = classifier.predict(X_test)
        inference_time = time.time() - start_time
        
        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.evaluate(test_labels, predictions, 'svm_embeddings')
        
        return SimulationResult(
            simulation_id=simulation_id,
            model_name='svm_embeddings',
            random_seed=random_seed,
            accuracy=metrics['accuracy'],
            precision_macro=metrics['precision_macro'],
            recall_macro=metrics['recall_macro'],
            f1_macro=metrics['f1_macro'],
            f1_weighted=metrics['f1_weighted'],
            training_time=training_time,
            inference_time=inference_time
        )
    
    def _run_bert(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
        val_labels: np.ndarray,
        test_texts: List[str],
        test_labels: np.ndarray,
        simulation_id: int,
        random_seed: int
    ) -> SimulationResult:
        """Run BERT model."""
        # BERT uses raw text (no preprocessing)
        
        # Training
        start_time = time.time()
        classifier = BERTClassifier(
            model_name=self.config.bert_model,
            max_length=self.config.bert_max_length,
            batch_size=self.config.bert_batch_size,
            learning_rate=self.config.bert_learning_rate,
            num_epochs=self.config.bert_epochs
        )
        classifier.fit(
            train_texts, train_labels.tolist(),
            val_texts, val_labels.tolist()
        )
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        predictions = classifier.predict(test_texts)
        inference_time = time.time() - start_time
        
        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.evaluate(test_labels, predictions, 'bert')
        
        return SimulationResult(
            simulation_id=simulation_id,
            model_name='bert',
            random_seed=random_seed,
            accuracy=metrics['accuracy'],
            precision_macro=metrics['precision_macro'],
            recall_macro=metrics['recall_macro'],
            f1_macro=metrics['f1_macro'],
            f1_weighted=metrics['f1_weighted'],
            training_time=training_time,
            inference_time=inference_time
        )
    
    def run_simulations(
        self,
        model_names: List[str],
        base_seed: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Run N simulations for each specified model.
        
        This method runs multiple simulations with different random seeds
        for each model and saves the results to CSV files.
        
        Args:
            model_names: List of model names to run simulations for
                        (e.g., ['svm_bow', 'svm_embeddings', 'bert'])
            base_seed: Base random seed (default 42). Each simulation uses
                      base_seed + simulation_id as its seed.
        
        Returns:
            Dictionary mapping model names to DataFrames with simulation results
            
        Validates: Requirements 7.1, 7.2, 7.8
        """
        logger.info(f"Starting simulations for models: {model_names}")
        logger.info(f"Number of simulations per model: {self.config.num_simulations}")
        
        all_results = {}
        
        for model_name in model_names:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running simulations for {model_name}")
            logger.info(f"{'='*80}")
            
            results = []
            
            # Run N simulations with different seeds
            for sim_id in tqdm(
                range(self.config.num_simulations),
                desc=f"{model_name} simulations"
            ):
                # Generate unique seed for this simulation
                random_seed = base_seed + sim_id
                
                try:
                    result = self.run_single_simulation(
                        model_name=model_name,
                        simulation_id=sim_id,
                        random_seed=random_seed
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(
                        f"Error in simulation {sim_id} for {model_name}: {e}",
                        exc_info=True
                    )
                    # Continue with next simulation
                    continue
            
            # Convert results to DataFrame
            if results:
                df = pd.DataFrame([asdict(r) for r in results])
                all_results[model_name] = df
                
                # Save to CSV
                output_file = self.output_dir / f"{model_name}_simulations.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved results to: {output_file}")
                
                # Log summary statistics
                self._log_summary_statistics(model_name, df)
            else:
                logger.warning(f"No successful simulations for {model_name}")
        
        logger.info(f"\n{'='*80}")
        logger.info("All simulations complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")
        
        return all_results
    
    def _log_summary_statistics(self, model_name: str, df: pd.DataFrame) -> None:
        """
        Log summary statistics for a model's simulations.
        
        Args:
            model_name: Name of the model
            df: DataFrame with simulation results
        """
        logger.info(f"\nSummary statistics for {model_name}:")
        logger.info(f"  Simulations completed: {len(df)}")
        
        # Calculate mean and std for key metrics
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
        
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                min_val = df[metric].min()
                max_val = df[metric].max()
                logger.info(
                    f"  {metric}: {mean_val:.4f} ± {std_val:.4f} "
                    f"(min={min_val:.4f}, max={max_val:.4f})"
                )
        
        # Log timing information
        if 'training_time' in df.columns:
            mean_train = df['training_time'].mean()
            logger.info(f"  Average training time: {mean_train:.2f}s")
        
        if 'inference_time' in df.columns:
            mean_infer = df['inference_time'].mean()
            logger.info(f"  Average inference time: {mean_infer:.2f}s")
    
    def calculate_confidence_intervals(
        self,
        df: pd.DataFrame,
        metric: str = 'f1_macro',
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate mean and confidence interval for a metric.
        
        Args:
            df: DataFrame with simulation results
            metric: Metric name (default 'f1_macro')
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        from scipy import stats
        
        values = df[metric].values
        mean = np.mean(values)
        std_err = stats.sem(values)
        
        # Calculate confidence interval
        ci = std_err * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        
        return mean, mean - ci, mean + ci
    
    def get_summary_table(
        self,
        results: Dict[str, pd.DataFrame],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate summary table with mean ± std for all models.
        
        Args:
            results: Dictionary mapping model names to result DataFrames
            metrics: List of metrics to include (default: key metrics)
            
        Returns:
            DataFrame with summary statistics
        """
        if metrics is None:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
        
        summary_data = []
        
        for model_name, df in results.items():
            row = {'model': model_name}
            
            for metric in metrics:
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            
            # Add timing information
            if 'training_time' in df.columns:
                row['training_time'] = f"{df['training_time'].mean():.2f}s"
            if 'inference_time' in df.columns:
                row['inference_time'] = f"{df['inference_time'].mean():.2f}s"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def run_all_simulations(
    config: ExperimentConfig,
    models: Optional[List[str]] = None,
    base_seed: int = 42,
    output_dir: str = 'results/simulations'
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run simulations for all models.
    
    Args:
        config: Experiment configuration
        models: List of model names (default: all models)
        base_seed: Base random seed (default 42)
        output_dir: Output directory for results
        
    Returns:
        Dictionary mapping model names to result DataFrames
    """
    if models is None:
        models = ['svm_bow', 'svm_embeddings', 'bert']
    
    runner = SimulationRunner(config, output_dir)
    results = runner.run_simulations(models, base_seed)
    
    # Generate and save summary table
    summary_table = runner.get_summary_table(results)
    summary_file = Path(output_dir) / 'summary_statistics.csv'
    summary_table.to_csv(summary_file, index=False)
    logger.info(f"\nSummary table saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_table.to_string(index=False))
    print("="*80 + "\n")
    
    return results
