"""
Demonstration of the Visualizer module with real simulation data.

This script shows how to use the Visualizer class to generate
professional visualizations for the sentiment analysis project.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from visualizer import Visualizer
from statistical_validator import StatisticalValidator

def main():
    """Demonstrate visualizer functionality with real data."""
    print("="*80)
    print("VISUALIZER DEMONSTRATION")
    print("="*80)
    
    # Initialize visualizer
    print("\n1. Initializing Visualizer...")
    viz = Visualizer(figsize=(10, 6))
    print("✓ Visualizer initialized")
    
    # Load simulation data
    print("\n2. Loading simulation data...")
    try:
        svm_bow_df = pd.read_csv('results/simulations/svm_bow_simulations.csv')
        svm_emb_df = pd.read_csv('results/simulations/svm_embeddings_simulations.csv')
        bert_df = pd.read_csv('results/simulations/bert_simulations.csv')
        
        # Fix BERT data format
        if 'model_name' not in bert_df.columns:
            bert_df['model_name'] = 'bert'
        if 'f1_score' in bert_df.columns and 'f1_macro' not in bert_df.columns:
            bert_df['f1_macro'] = bert_df['f1_score']
        
        # Combine all simulations
        all_sims = pd.concat([svm_bow_df, svm_emb_df, bert_df], ignore_index=True)
        all_sims = all_sims.dropna(subset=['model_name'])
        
        print(f"✓ Loaded {len(all_sims)} simulation results")
        print(f"  Models: {all_sims['model_name'].unique().tolist()}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # 1. Metrics Comparison
    print("\n3. Generating metrics comparison plot...")
    models = all_sims['model_name'].unique().tolist()
    results = {}
    for model in models:
        model_data = all_sims[all_sims['model_name'] == model]
        results[model] = {
            'accuracy': float(model_data['accuracy'].mean()),
            'f1_macro': float(model_data['f1_macro'].mean()),
            'precision_macro': float(model_data.get('precision_macro', model_data.get('precision', 0)).mean()),
            'recall_macro': float(model_data.get('recall_macro', model_data.get('recall', 0)).mean())
        }
    
    viz.plot_metrics_comparison(results, metrics=['accuracy', 'f1_macro'])
    print("✓ Metrics comparison plot created")
    
    # 2. Confusion Matrices (sample data)
    print("\n4. Generating confusion matrix plots...")
    # Note: In real usage, these would come from actual model predictions
    sample_cms = {
        'svm_bow': np.array([[450, 50], [30, 470]]),
        'svm_embeddings': np.array([[460, 40], [35, 465]]),
        'bert': np.array([[480, 20], [25, 475]])
    }
    
    for model, cm in sample_cms.items():
        if model in models:
            viz.plot_confusion_matrix(cm, model)
    print(f"✓ Generated {len(sample_cms)} confusion matrix plots")
    
    # 3. Boxplots
    print("\n5. Generating boxplot for metric distribution...")
    viz.plot_boxplots(all_sims, metric='f1_macro')
    viz.plot_boxplots(all_sims, metric='accuracy')
    print("✓ Boxplots created")
    
    # 4. Line Evolution
    print("\n6. Generating line evolution plots...")
    viz.plot_line_evolution(all_sims, metric='f1_macro')
    viz.plot_line_evolution(all_sims, metric='accuracy')
    print("✓ Line evolution plots created")
    
    # 5. P-value Matrix
    print("\n7. Generating p-value matrix...")
    try:
        validator = StatisticalValidator(alpha=0.05)
        
        # Generate p-value matrix for f1_macro
        pvalue_matrix = validator.generate_pvalue_matrix(models, metric='f1_macro')
        viz.plot_pvalue_matrix(pvalue_matrix, models, metric='f1_macro')
        
        print("✓ P-value matrix created")
    except Exception as e:
        print(f"⚠ Could not generate p-value matrix: {e}")
        print("  Creating sample p-value matrix instead...")
        n_models = len(models)
        pvalue_matrix = np.random.uniform(0, 0.1, (n_models, n_models))
        np.fill_diagonal(pvalue_matrix, 1.0)
        viz.plot_pvalue_matrix(pvalue_matrix, models, metric='f1_macro')
        print("✓ Sample p-value matrix created")
    
    # 6. Confidence Intervals
    print("\n8. Generating confidence interval plots...")
    ci_results = {}
    for model in models:
        model_data = all_sims[all_sims['model_name'] == model]
        ci_results[model] = {
            'mean': float(model_data['f1_macro'].mean()),
            'std': float(model_data['f1_macro'].std())
        }
    
    viz.plot_confidence_intervals(ci_results, metric='f1_macro')
    print("✓ Confidence interval plot created")
    
    # 7. Save All Figures
    print("\n9. Saving all figures...")
    viz.save_all_figures(output_dir='results/plots', dpi=300)
    print("✓ All figures saved to results/plots/")
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(viz.figures)} visualizations:")
    for fig_name in viz.figures.keys():
        print(f"  - {fig_name}.png")
    print("\nAll plots saved to: results/plots/")
    print("Resolution: 300 DPI (high quality for presentations)")
    
    # Clean up
    viz.close_all()

if __name__ == "__main__":
    main()
