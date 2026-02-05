"""
Script to create perturbed test datasets for robustness analysis.

This script loads the test dataset and creates a perturbed version with typos
for testing model robustness.

Usage:
    python scripts/create_perturbed_dataset.py --seed 42 --rate 0.05
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append('.')

from src.data_loader import DataLoader
from src.text_perturbation import TextPerturbation
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to create perturbed dataset."""
    parser = argparse.ArgumentParser(
        description='Create perturbed test dataset for robustness analysis'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data loading (default: 42)'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=0.05,
        help='Perturbation rate - probability of perturbing each word (default: 0.05)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/perturbed/test',
        help='Output directory for perturbed dataset (default: data/perturbed/test)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("CREATING PERTURBED TEST DATASET")
    logger.info("="*80)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Perturbation rate: {args.rate}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load test data
    logger.info("\nLoading test data...")
    data_loader = DataLoader(random_state=args.seed)
    train_df, val_df, test_df = data_loader.load()
    
    logger.info(f"Test set size: {len(test_df)} samples")
    
    # Create perturbation object
    logger.info(f"\nCreating perturbations with rate={args.rate}...")
    perturber = TextPerturbation(
        perturbation_rate=args.rate,
        random_state=args.seed
    )
    
    # Perturb test texts
    test_texts = test_df['text'].tolist()
    perturbed_texts = perturber.perturb_texts(test_texts)
    
    # Create perturbed DataFrame
    perturbed_df = test_df.copy()
    perturbed_df['text'] = perturbed_texts
    
    # Show examples
    logger.info("\nPerturbation examples:")
    logger.info("-"*80)
    examples = perturber.get_perturbation_examples(test_texts, n_examples=5)
    for i, (original, perturbed) in enumerate(examples, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Original:  {original[:100]}...")
        logger.info(f"  Perturbed: {perturbed[:100]}...")
    
    # Save perturbed dataset
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f'test_typos_seed{args.seed}.csv'
    perturbed_df.to_csv(output_file, index=False)
    
    logger.info("\n" + "="*80)
    logger.info(f"✓ Perturbed dataset saved to: {output_file}")
    logger.info(f"  Total samples: {len(perturbed_df)}")
    logger.info(f"  Perturbation rate: {args.rate}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
