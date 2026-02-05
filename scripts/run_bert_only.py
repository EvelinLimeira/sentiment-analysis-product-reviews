"""
Run BERT simulations only.

This script runs only BERT simulations, useful for running in parallel
with other model simulations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.simulation_runner import SimulationRunner


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run BERT simulations only')
    parser.add_argument('--num-simulations', type=int, default=30,
                       help='Number of simulations (default: 30)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from simulation N (default: 0)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BERT SIMULATIONS ONLY - IMPROVED TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Total simulations: {args.num_simulations}")
    print(f"  Starting from: {args.start_from}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  BERT epochs: 10 (with early stopping)")
    print(f"  BERT batch size: 32")
    print(f"  Early stopping patience: 3")
    print()
    
    # Use full config with improved BERT settings
    config = ExperimentConfig(
        dataset_name='amazon_reviews',
        num_simulations=args.num_simulations
    )
    
    print(f"BERT Configuration:")
    print(f"  Model: {config.bert_model}")
    print(f"  Max length: {config.bert_max_length}")
    print(f"  Batch size: {config.bert_batch_size}")
    print(f"  Epochs: {config.bert_epochs}")
    print(f"  Learning rate: {config.bert_learning_rate}")
    print()
    
    # Initialize runner
    runner = SimulationRunner(config, output_dir='results/simulations')
    
    # Run only BERT simulations
    print("Starting BERT simulations with improved training...")
    print("Expected time: ~3-5 minutes per simulation on GPU")
    print(f"Total estimated time: ~{args.num_simulations * 4} minutes")
    print()
    
    results = runner.run_simulations(
        model_names=['bert'],
        base_seed=args.base_seed
    )
    
    print("\n" + "=" * 80)
    print("BERT SIMULATIONS COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
