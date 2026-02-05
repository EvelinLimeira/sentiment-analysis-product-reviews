"""
Run multiple simulations for statistical validation.

This script runs N simulations (default 10) for each classifier:
- SVM + Bag of Words
- SVM + Embeddings
- BERT (optional, requires GPU)

Each simulation uses a different random seed to ensure statistical validity.
Results are saved to results/simulations/ for later analysis.
"""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.simulation_runner import SimulationRunner
from src.statistical_validator import StatisticalValidator
import pandas as pd

# Reduce logging verbosity for noisy modules
logging.getLogger('src.data_loader').setLevel(logging.WARNING)
logging.getLogger('src.preprocessor').setLevel(logging.WARNING)
logging.getLogger('src.embedding_encoder').setLevel(logging.WARNING)
logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='Run multiple simulations for statistical validation')
    parser.add_argument('--num-simulations', type=int, default=10,
                       help='Number of simulations per model (default: 10)')
    parser.add_argument('--models', nargs='+', 
                       default=['svm_bow', 'svm_embeddings'],
                       choices=['svm_bow', 'svm_embeddings', 'bert'],
                       help='Models to evaluate (default: svm_bow svm_embeddings)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick test config (reduced parameters)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MULTIPLE SIMULATIONS FOR STATISTICAL VALIDATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Number of simulations: {args.num_simulations}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Quick mode: {args.quick}")
    print()
    
    # Create config
    if args.quick:
        from src.config import get_quick_test_config
        config = get_quick_test_config()
        config.num_simulations = args.num_simulations
    else:
        config = ExperimentConfig(
            dataset_name='amazon_reviews',
            num_simulations=args.num_simulations
        )
    
    # Initialize runner
    runner = SimulationRunner(config, output_dir='results/simulations')
    
    # Run simulations
    print(f"Starting {args.num_simulations} simulations for each model...")
    print("This may take a while depending on the number of simulations and models.")
    print()
    
    results = runner.run_simulations(
        model_names=args.models,
        base_seed=args.base_seed
    )
    
    # Display summary for each model
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
            mean = df[metric].mean()
            std = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            print(f"{metric:<20} {mean:.4f}    {std:.4f}    {min_val:.4f}    {max_val:.4f}")
        
        # Timing
        print(f"\nTiming:")
        print(f"  Training time:   {df['training_time'].mean():.2f}s (±{df['training_time'].std():.2f}s)")
        print(f"  Inference time:  {df['inference_time'].mean():.2f}s (±{df['inference_time'].std():.2f}s)")
        
        # 95% Confidence Intervals
        print(f"\n95% Confidence Intervals:")
        for metric in ['accuracy', 'f1_macro']:
            mean, lower, upper = runner.calculate_confidence_intervals(df, metric=metric)
            print(f"  {metric:<15} {mean:.4f} [{lower:.4f}, {upper:.4f}]")
    
    # Statistical comparison if multiple models
    if len(args.models) > 1:
        print("\n" + "=" * 80)
        print("STATISTICAL COMPARISON")
        print("=" * 80)
        
        validator = StatisticalValidator(alpha=0.05)
        
        # Pairwise comparisons
        models = list(results.keys())
        print("\nPairwise Wilcoxon Tests (F1 Macro):")
        print("-" * 80)
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1 = models[i]
                model2 = models[j]
                
                # Get F1 scores
                scores1 = results[model1]['f1_macro'].values
                scores2 = results[model2]['f1_macro'].values
                
                # Perform test
                result = validator.wilcoxon_test(scores1, scores2)
                
                winner = model1 if result['mean_diff'] > 0 else model2
                significant = "✓" if result['significant'] else "✗"
                
                print(f"\n{model1} vs {model2}:")
                print(f"  p-value: {result['p_value']:.4f} {significant}")
                print(f"  Mean difference: {result['mean_diff']:.4f}")
                if result['significant']:
                    print(f"  Winner: {winner}")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: results/simulations/")
    print(f"Files:")
    for model_name in args.models:
        print(f"  - {model_name}_simulations.csv")
    
    return 0


if __name__ == "__main__":
    exit(main())
