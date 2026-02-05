"""
Run BERT simulations only.

This script runs only BERT simulations, useful for running in parallel
with other model simulations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_quick_test_config
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
    print("BERT SIMULATIONS ONLY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Total simulations: {args.num_simulations}")
    print(f"  Starting from: {args.start_from}")
    print(f"  Base seed: {args.base_seed}")
    print()
    
    # Use quick config for faster training
    config = get_quick_test_config()
    config.num_simulations = args.num_simulations
    
    # Initialize runner
    runner = SimulationRunner(config, output_dir='results/simulations')
    
    # Run only BERT simulations
    print("Starting BERT simulations with GPU acceleration...")
    print("This will take approximately 1-2 hours for 30 simulations.")
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
