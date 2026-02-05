"""
Demo script for advanced NLP analysis.

This script demonstrates the complete advanced analysis workflow including:
1. Loading test data and trained models
2. Generating predictions
3. Creating perturbed dataset
4. Running all advanced analyses
5. Generating visualizations

Usage:
    python scripts/demo_advanced_analysis.py --seed 42
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path
sys.path.append('.')

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.vectorizers import BoWVectorizer
from src.embedding_encoder import EmbeddingEncoder
from src.classifiers import SVMClassifier
from src.bert_classifier import BERTClassifier
from src.advanced_analysis import AdvancedNLPAnalysis
from src.text_perturbation import TextPerturbation
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_predict(model_name: str, test_df, seed: int):
    """
    Load a trained model and generate predictions.
    
    Args:
        model_name: Name of the model ('svm_bow', 'svm_embeddings', 'bert')
        test_df: Test DataFrame
        seed: Random seed
        
    Returns:
        Predictions array or None if model not found
    """
    model_dir = Path(f'results/models/{model_name}')
    
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return None
    
    try:
        test_texts = test_df['text'].tolist()
        
        if model_name == 'svm_bow':
            logger.info(f"Loading {model_name}...")
            preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
            vectorizer = joblib.load(model_dir / 'vectorizer.pkl')
            classifier = joblib.load(model_dir / 'classifier.pkl')
            
            # Preprocess and predict
            test_texts_processed = preprocessor.transform(test_texts)
            X_test = vectorizer.transform(test_texts_processed)
            predictions = classifier.predict(X_test)
            
        elif model_name == 'svm_embeddings':
            logger.info(f"Loading {model_name}...")
            preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
            encoder = joblib.load(model_dir / 'encoder.pkl')
            classifier = joblib.load(model_dir / 'classifier.pkl')
            
            # Preprocess and predict
            test_texts_processed = preprocessor.transform(test_texts)
            X_test = encoder.encode_batch(test_texts_processed)
            predictions = classifier.predict(X_test)
            
        elif model_name == 'bert':
            logger.info(f"Loading {model_name}...")
            bert_model_dir = model_dir / 'bert_model'
            classifier = BERTClassifier.load_model(str(bert_model_dir))
            
            # Predict
            predictions = classifier.predict(test_texts)
        
        else:
            logger.error(f"Unknown model name: {model_name}")
            return None
        
        logger.info(f"✓ Generated predictions for {model_name}")
        return predictions
        
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}", exc_info=True)
        return None


def main():
    """Main function for advanced analysis demo."""
    parser = argparse.ArgumentParser(
        description='Demo script for advanced NLP analysis'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--perturbation-rate',
        type=float,
        default=0.05,
        help='Perturbation rate for typo analysis (default: 0.05)'
    )
    parser.add_argument(
        '--n-sarcasm',
        type=int,
        default=50,
        help='Number of sarcastic samples to detect (default: 50)'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("ADVANCED NLP ANALYSIS DEMO")
    logger.info("="*80)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Perturbation rate: {args.perturbation_rate}")
    
    # Step 1: Load test data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading test data")
    logger.info("="*80)
    
    data_loader = DataLoader(random_state=args.seed)
    train_df, val_df, test_df = data_loader.load()
    logger.info(f"✓ Test set size: {len(test_df)} samples")
    
    # Step 2: Load models and generate predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Loading models and generating predictions")
    logger.info("="*80)
    
    predictions = {}
    for model_name in ['svm_bow', 'svm_embeddings', 'bert']:
        preds = load_model_and_predict(model_name, test_df, args.seed)
        if preds is not None:
            predictions[model_name] = preds
    
    if not predictions:
        logger.error("No models loaded. Please train models first.")
        return
    
    logger.info(f"\n✓ Loaded {len(predictions)} models: {list(predictions.keys())}")
    
    # Step 3: Create perturbed dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Creating perturbed dataset for typo analysis")
    logger.info("="*80)
    
    perturber = TextPerturbation(
        perturbation_rate=args.perturbation_rate,
        random_state=args.seed
    )
    
    test_texts = test_df['text'].tolist()
    perturbed_texts = perturber.perturb_texts(test_texts)
    
    # Show examples
    examples = perturber.get_perturbation_examples(test_texts, n_examples=3)
    logger.info("\nPerturbation examples:")
    for i, (orig, pert) in enumerate(examples, 1):
        logger.info(f"\n  Example {i}:")
        logger.info(f"    Original:  {orig[:80]}...")
        logger.info(f"    Perturbed: {pert[:80]}...")
    
    # Generate predictions on perturbed data
    logger.info("\nGenerating predictions on perturbed data...")
    perturbed_predictions = {}
    
    # Create temporary DataFrame with perturbed texts
    perturbed_df = test_df.copy()
    perturbed_df['text'] = perturbed_texts
    
    for model_name in predictions.keys():
        preds = load_model_and_predict(model_name, perturbed_df, args.seed)
        if preds is not None:
            perturbed_predictions[model_name] = preds
    
    logger.info(f"✓ Generated perturbed predictions for {len(perturbed_predictions)} models")
    
    # Step 4: Detect potential sarcasm
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Detecting potential sarcastic reviews")
    logger.info("="*80)
    
    # Simple heuristic-based detection
    positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect']
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'bad']
    
    sarcasm_indices = []
    for idx, row in test_df.iterrows():
        text_lower = row['text'].lower()
        label = row['label']
        
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_words)
        
        # Negative review with positive words or vice versa
        if (label == 0 and has_positive) or (label == 1 and has_negative):
            sarcasm_indices.append(idx)
        
        if len(sarcasm_indices) >= args.n_sarcasm:
            break
    
    logger.info(f"✓ Detected {len(sarcasm_indices)} potentially sarcastic reviews")
    
    # Show examples
    if sarcasm_indices:
        logger.info("\nExamples of potentially sarcastic reviews:")
        for i, idx in enumerate(sarcasm_indices[:3], 1):
            row = test_df.loc[idx]
            logger.info(f"\n  Example {i}:")
            logger.info(f"    Text: {row['text'][:80]}...")
            logger.info(f"    Label: {'Positive' if row['label'] == 1 else 'Negative'}")
    
    # Step 5: Run advanced analyses
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Running advanced NLP analyses")
    logger.info("="*80)
    
    analyzer = AdvancedNLPAnalysis(test_df, predictions)
    
    results = analyzer.run_all_analyses(
        perturbed_predictions=perturbed_predictions if perturbed_predictions else None,
        sarcasm_indices=sarcasm_indices if sarcasm_indices else None
    )
    
    # Step 6: Summary
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nResults saved to:")
    logger.info(f"  - CSV files: {analyzer.output_dir}")
    logger.info(f"  - Plots: {analyzer.plots_dir}")
    logger.info(f"\nAnalyses completed:")
    for analysis_name in results.keys():
        logger.info(f"  ✓ {analysis_name.replace('_', ' ').title()}")
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
