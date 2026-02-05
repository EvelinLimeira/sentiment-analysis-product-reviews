"""
Script to help manually annotate sarcastic reviews.

This script presents test samples to the user for manual annotation of sarcasm.
It saves the indices of sarcastic reviews for later analysis.

Usage:
    python scripts/annotate_sarcasm.py --seed 42 --n-samples 100
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path
sys.path.append('.')

from src.data_loader import DataLoader
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def annotate_sarcasm_interactive(test_df: pd.DataFrame, n_samples: int = 100) -> list:
    """
    Interactively annotate sarcastic reviews.
    
    Args:
        test_df: Test DataFrame
        n_samples: Number of samples to annotate
        
    Returns:
        List of indices of sarcastic reviews
    """
    sarcasm_indices = []
    
    # Sample reviews for annotation
    sample_size = min(n_samples, len(test_df))
    sampled_df = test_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("SARCASM ANNOTATION")
    print("="*80)
    print(f"Annotating {sample_size} reviews")
    print("For each review, type 'y' if sarcastic, 'n' if not, 'q' to quit")
    print("="*80 + "\n")
    
    for idx, row in sampled_df.iterrows():
        original_idx = test_df[test_df['text'] == row['text']].index[0]
        
        print(f"\n[{idx + 1}/{sample_size}] Review (Index: {original_idx}):")
        print("-"*80)
        print(f"Text: {row['text']}")
        print(f"Label: {'Positive' if row['label'] == 1 else 'Negative'}")
        print("-"*80)
        
        while True:
            response = input("Is this sarcastic? (y/n/q): ").strip().lower()
            
            if response == 'q':
                print("\nAnnotation stopped by user.")
                return sarcasm_indices
            elif response == 'y':
                sarcasm_indices.append(int(original_idx))
                print("✓ Marked as sarcastic")
                break
            elif response == 'n':
                print("✓ Marked as not sarcastic")
                break
            else:
                print("Invalid input. Please enter 'y', 'n', or 'q'.")
    
    return sarcasm_indices


def auto_detect_potential_sarcasm(test_df: pd.DataFrame, n_samples: int = 100) -> list:
    """
    Automatically detect potential sarcastic reviews using heuristics.
    
    This is a simple heuristic-based approach that looks for:
    - Positive words in negative reviews
    - Negative words in positive reviews
    - Excessive punctuation (!!!, ???)
    - Quotation marks around sentiment words
    
    Args:
        test_df: Test DataFrame
        n_samples: Maximum number of potential sarcastic samples to return
        
    Returns:
        List of indices of potentially sarcastic reviews
    """
    logger.info("Auto-detecting potential sarcasm using heuristics...")
    
    # Define sentiment words
    positive_words = [
        'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect',
        'love', 'best', 'awesome', 'brilliant', 'outstanding'
    ]
    negative_words = [
        'terrible', 'awful', 'horrible', 'worst', 'hate', 'bad', 'poor',
        'disappointing', 'useless', 'waste'
    ]
    
    potential_sarcasm = []
    
    for idx, row in test_df.iterrows():
        text_lower = row['text'].lower()
        label = row['label']
        
        # Check for contradictions
        has_positive_words = any(word in text_lower for word in positive_words)
        has_negative_words = any(word in text_lower for word in negative_words)
        
        # Negative review with positive words (potential sarcasm)
        if label == 0 and has_positive_words:
            potential_sarcasm.append(idx)
        # Positive review with negative words (potential sarcasm)
        elif label == 1 and has_negative_words:
            potential_sarcasm.append(idx)
        # Excessive punctuation
        elif '!!!' in row['text'] or '???' in row['text']:
            potential_sarcasm.append(idx)
        # Quotation marks (often used for sarcasm)
        elif '"' in row['text'] and (has_positive_words or has_negative_words):
            potential_sarcasm.append(idx)
    
    # Limit to n_samples
    if len(potential_sarcasm) > n_samples:
        import random
        random.seed(42)
        potential_sarcasm = random.sample(potential_sarcasm, n_samples)
    
    logger.info(f"Found {len(potential_sarcasm)} potentially sarcastic reviews")
    
    return potential_sarcasm


def main():
    """Main function for sarcasm annotation."""
    parser = argparse.ArgumentParser(
        description='Annotate sarcastic reviews in test dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data loading (default: 42)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of samples to annotate (default: 100)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'auto'],
        default='auto',
        help='Annotation mode: interactive (manual) or auto (heuristic-based) (default: auto)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/advanced_analysis/sarcasm_indices.json',
        help='Output file for sarcasm indices (default: results/advanced_analysis/sarcasm_indices.json)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SARCASM ANNOTATION")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Samples: {args.n_samples}")
    
    # Load test data
    logger.info("\nLoading test data...")
    data_loader = DataLoader(random_state=args.seed)
    train_df, val_df, test_df = data_loader.load()
    
    logger.info(f"Test set size: {len(test_df)} samples")
    
    # Annotate sarcasm
    if args.mode == 'interactive':
        sarcasm_indices = annotate_sarcasm_interactive(test_df, args.n_samples)
    else:  # auto
        sarcasm_indices = auto_detect_potential_sarcasm(test_df, args.n_samples)
        
        # Show examples
        logger.info("\nExamples of potentially sarcastic reviews:")
        logger.info("-"*80)
        for i, idx in enumerate(sarcasm_indices[:5], 1):
            row = test_df.loc[idx]
            logger.info(f"\nExample {i} (Index: {idx}):")
            logger.info(f"  Text: {row['text'][:100]}...")
            logger.info(f"  Label: {'Positive' if row['label'] == 1 else 'Negative'}")
    
    # Save indices
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'sarcasm_indices': sarcasm_indices,
            'n_samples': len(sarcasm_indices),
            'seed': args.seed,
            'mode': args.mode
        }, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info(f"✓ Sarcasm indices saved to: {output_path}")
    logger.info(f"  Total sarcastic samples: {len(sarcasm_indices)}")
    logger.info(f"  Percentage: {len(sarcasm_indices)/len(test_df)*100:.2f}%")
    logger.info("="*80)


if __name__ == "__main__":
    main()
