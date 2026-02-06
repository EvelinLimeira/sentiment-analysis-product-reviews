"""
Run simulations with timestamped output to avoid overwriting previous results.

This script saves results with timestamps so you can keep multiple runs.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.simulation_runner import SimulationRunner
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Run simulations with timestamped output')
    parser.add_argument('--num-simulations', type=int, default=30,
                       help='Number of simulations per model (default: 30)')
    parser.add_argument('--models', nargs='+', 
                       default=['bert'],
                       choices=['svm_bow', 'svm_embeddings', 'bert'],
                       help='Models to evaluate (default: bert)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom run name (default: timestamp)')
    
    args = parser.parse_args()
    
    # Generate run identifier
    if args.run_name:
        run_id = args.run_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"run_{timestamp}"
    
    # Create output directory with run ID
    output_dir = f'results/simulations/{run_id}'
    
    print("=" * 80)
    print("SIMULATIONS WITH TIMESTAMPED OUTPUT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Run ID: {run_id}")
    print(f"  Number of simulations: {args.num_simulations}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Create config
    config = ExperimentConfig(
        dataset_name='amazon_reviews',
        num_simulations=args.num_simulations
    )
    
    # Initialize runner with custom output directory
    runner = SimulationRunner(config, output_dir=output_dir)
    
    # Run simulations
    print(f"Starting {args.num_simulations} simulations for each model...")
    print("Results will be saved with unique names to avoid overwriting.")
    print()
    
    results = runner.run_simulations(
        model_names=args.models,
        base_seed=args.base_seed
    )
    
    # Display summary
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)
    
    for model_name, df in results.items():
        print(f"\n{model_name.upper().replace('_', ' + ')}:")
        print("-" * 80)
        
        # Calculate statistics
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
        
        print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)
        
        for metric in metrics:
            if metric in df.columns:
                mean = df[metric].mean()
                std = df[metric].std()
                min_val = df[metric].min()
                max_val = df[metric].max()
                print(f"{metric:<20} {mean:.4f}    {std:.4f}    {min_val:.4f}    {max_val:.4f}")
        
        # 95% Confidence Intervals
        print(f"\n95% Confidence Intervals:")
        for metric in ['accuracy', 'f1_macro']:
            if metric in df.columns:
                mean, lower, upper = runner.calculate_confidence_intervals(df, metric=metric)
                print(f"  {metric:<15} {mean:.4f} [{lower:.4f}, {upper:.4f}]")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Run ID: {run_id}")
    print("\nFiles:")
    for model_name in args.models:
        print(f"  - {run_id}/{model_name}_simulations.csv")
    print()
    
    # Save run metadata
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'num_simulations': args.num_simulations,
        'models': args.models,
        'base_seed': args.base_seed,
        'output_dir': output_dir
    }
    
    import json
    metadata_file = Path(output_dir) / 'run_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
