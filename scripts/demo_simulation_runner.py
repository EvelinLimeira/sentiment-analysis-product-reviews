"""
Demo script for the simulation runner.

This script demonstrates how to use the SimulationRunner to run multiple
simulations for statistical validation. It runs a quick test with 2 simulations
per model to verify the implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ExperimentConfig, get_quick_test_config
from src.simulation_runner import SimulationRunner, run_all_simulations


def demo_single_model():
    """Demo: Run simulations for a single model."""
    print("="*80)
    print("DEMO: Single Model Simulations (SVM + BoW)")
    print("="*80)
    
    # Create quick test config (2 simulations)
    config = get_quick_test_config()
    
    # Initialize runner
    runner = SimulationRunner(config, output_dir='results/simulations')
    
    # Run simulations for SVM + BoW only
    results = runner.run_simulations(
        model_names=['svm_bow'],
        base_seed=42
    )
    
    # Display results
    print("\nResults DataFrame:")
    print(results['svm_bow'])
    
    # Calculate confidence intervals
    print("\n95% Confidence Intervals:")
    for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
        mean, lower, upper = runner.calculate_confidence_intervals(
            results['svm_bow'],
            metric=metric
        )
        print(f"  {metric}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")


def demo_all_models():
    """Demo: Run simulations for all models."""
    print("\n" + "="*80)
    print("DEMO: All Models Simulations")
    print("="*80)
    
    # Create quick test config (2 simulations)
    config = get_quick_test_config()
    
    # Run simulations for all models
    results = run_all_simulations(
        config=config,
        models=['svm_bow', 'svm_embeddings', 'bert'],
        base_seed=42,
        output_dir='results/simulations'
    )
    
    print("\nSimulation results saved to: results/simulations/")
    print("Files created:")
    for model_name in results.keys():
        print(f"  - {model_name}_simulations.csv")
    print("  - summary_statistics.csv")


def demo_custom_simulations():
    """Demo: Run custom number of simulations."""
    print("\n" + "="*80)
    print("DEMO: Custom Simulations (5 runs)")
    print("="*80)
    
    # Create custom config with 5 simulations
    config = ExperimentConfig(
        dataset_name='amazon_reviews',
        num_simulations=5,
        tfidf_max_features=1000,  # Reduced for speed
        bert_epochs=1,  # Reduced for speed
        bert_batch_size=8
    )
    
    # Run simulations for SVM models only (faster than BERT)
    runner = SimulationRunner(config, output_dir='results/simulations')
    results = runner.run_simulations(
        model_names=['svm_bow', 'svm_embeddings'],
        base_seed=100
    )
    
    # Generate summary table
    summary = runner.get_summary_table(results)
    print("\nSummary Table:")
    print(summary.to_string(index=False))


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("SIMULATION RUNNER DEMO")
    print("="*80)
    print("\nThis demo shows how to use the SimulationRunner for statistical validation.")
    print("It will run quick tests with reduced parameters for demonstration.\n")
    
    # Demo 1: Single model
    demo_single_model()
    
    # Demo 2: All models (commented out by default - takes longer)
    # Uncomment to run full demo
    # demo_all_models()
    
    # Demo 3: Custom simulations
    # demo_custom_simulations()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nTo run full simulations (10+ runs per model):")
    print("  1. Create a config with num_simulations=10 (or more)")
    print("  2. Use run_all_simulations() with your config")
    print("  3. Results will be saved to results/simulations/")
    print("  4. Use StatisticalValidator to analyze the results")
    print("\nExample:")
    print("  from src.config import get_default_config")
    print("  from src.simulation_runner import run_all_simulations")
    print("  config = get_default_config()")
    print("  results = run_all_simulations(config)")
    print()


if __name__ == '__main__':
    main()
