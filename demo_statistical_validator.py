"""
Demo script for StatisticalValidator module.

This script demonstrates how to use the StatisticalValidator class
to perform statistical significance tests on model comparison results.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from src.statistical_validator import StatisticalValidator


def create_sample_simulation_data():
    """Create sample simulation data for demonstration."""
    print("Creating sample simulation data...")
    
    # Ensure directories exist
    Path('results/simulations').mkdir(parents=True, exist_ok=True)
    Path('results/plots/statistical').mkdir(parents=True, exist_ok=True)
    Path('results/statistical_tests').mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Model 1: SVM with BoW (good performance)
    svm_bow_data = pd.DataFrame({
        'simulation_id': range(10),
        'accuracy': np.random.normal(0.85, 0.02, 10),
        'precision': np.random.normal(0.84, 0.02, 10),
        'recall': np.random.normal(0.86, 0.02, 10),
        'f1_score': np.random.normal(0.85, 0.02, 10),
        'training_time': np.random.normal(100, 10, 10),
        'inference_time': np.random.normal(5, 0.5, 10)
    })
    
    # Model 2: SVM with Embeddings (slightly worse)
    svm_emb_data = pd.DataFrame({
        'simulation_id': range(10),
        'accuracy': np.random.normal(0.80, 0.02, 10),
        'precision': np.random.normal(0.79, 0.02, 10),
        'recall': np.random.normal(0.81, 0.02, 10),
        'f1_score': np.random.normal(0.80, 0.02, 10),
        'training_time': np.random.normal(120, 10, 10),
        'inference_time': np.random.normal(6, 0.5, 10)
    })
    
    # Model 3: BERT (best performance)
    bert_data = pd.DataFrame({
        'simulation_id': range(10),
        'accuracy': np.random.normal(0.90, 0.015, 10),
        'precision': np.random.normal(0.89, 0.015, 10),
        'recall': np.random.normal(0.91, 0.015, 10),
        'f1_score': np.random.normal(0.90, 0.015, 10),
        'training_time': np.random.normal(500, 50, 10),
        'inference_time': np.random.normal(20, 2, 10)
    })
    
    # Save to CSV
    svm_bow_data.to_csv('results/simulations/svm_bow_simulations.csv', index=False)
    svm_emb_data.to_csv('results/simulations/svm_embeddings_simulations.csv', index=False)
    bert_data.to_csv('results/simulations/bert_simulations.csv', index=False)
    
    print("✓ Sample data created in results/simulations/")
    return ['svm_bow', 'svm_embeddings', 'bert']


def demo_statistical_validation():
    """Demonstrate statistical validation workflow."""
    print("\n" + "="*80)
    print("STATISTICAL VALIDATOR DEMO")
    print("="*80 + "\n")
    
    # Create sample data
    model_names = create_sample_simulation_data()
    
    # Initialize validator
    validator = StatisticalValidator(alpha=0.05)
    print(f"\n✓ StatisticalValidator initialized with α = {validator.alpha}")
    
    # 1. Test normality for each model
    print("\n" + "-"*80)
    print("1. SHAPIRO-WILK NORMALITY TESTS")
    print("-"*80)
    
    for model in model_names:
        result = validator.shapiro_normality(model, 'f1_score')
        print(f"\n{model}:")
        print(f"  Statistic: {result['statistic']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  Is Normal: {result['is_normal']}")
    
    # 2. Kruskal-Wallis test
    print("\n" + "-"*80)
    print("2. KRUSKAL-WALLIS TEST (Multiple Groups)")
    print("-"*80)
    
    kw_result = validator.kruskal_wallis_multiple(model_names, 'f1_score')
    print(f"\nMetric: F1-Score")
    print(f"  H statistic: {kw_result['statistic']:.4f}")
    print(f"  P-value: {kw_result['p_value']:.6f}")
    print(f"  Significant: {kw_result['significant']}")
    
    if kw_result['significant']:
        print("  → There IS significant difference between models")
    else:
        print("  → There is NO significant difference between models")
    
    # 3. Pairwise Wilcoxon tests
    print("\n" + "-"*80)
    print("3. WILCOXON PAIRWISE TESTS")
    print("-"*80)
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            result = validator.wilcoxon_pairwise(model1, model2, 'f1_score')
            print(f"\n{model1} vs {model2}:")
            print(f"  Median {model1}: {result['model1_median']:.4f}")
            print(f"  Median {model2}: {result['model2_median']:.4f}")
            print(f"  P-value: {result['p_value']:.6f}")
            print(f"  Significant: {result['significant']}")
            print(f"  Winner: {result['winner']}")
    
    # 4. Generate p-value matrix
    print("\n" + "-"*80)
    print("4. P-VALUE MATRIX")
    print("-"*80)
    
    matrix = validator.generate_pvalue_matrix(model_names, 'f1_score')
    print("\nP-value matrix generated:")
    print(matrix)
    print("\n✓ Heatmap saved to: results/plots/statistical/pvalue_matrix_f1_score.png")
    
    # 5. Generate complete report
    print("\n" + "-"*80)
    print("5. COMPLETE STATISTICAL REPORT")
    print("-"*80)
    
    metrics = ['accuracy', 'f1_score']
    report = validator.generate_report(model_names, metrics)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/simulations/*.csv (simulation data)")
    print("  - results/plots/statistical/pvalue_matrix_*.png (p-value heatmaps)")
    print("  - results/statistical_tests/statistical_report.txt (complete report)")
    print("\n")


if __name__ == "__main__":
    demo_statistical_validation()
