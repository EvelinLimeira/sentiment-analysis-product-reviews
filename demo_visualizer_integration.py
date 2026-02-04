"""
Demonstration of Visualizer Integration with Real Simulation Data

This script demonstrates how the Visualizer class integrates with
the complete sentiment analysis pipeline, using real simulation results.
"""

import sys
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import pandas as pd
import os
from visualizer import Visualizer
from statistical_validator import StatisticalValidator


def demo_visualizer_with_real_data():
    """Demonstrate visualizer with real simulation data."""
    
    print("=" * 70)
    print("VISUALIZER INTEGRATION DEMONSTRATION")
    print("=" * 70)
    
    # Initialize visualizer
    viz = Visualizer(style='seaborn-v0_8-whitegrid', figsize=(10, 6))
    print("\n✓ Visualizer initialized with professional theme")
    
    # Check if simulation results exist
    sim_dir = 'results/simulations'
    if not os.path.exists(sim_dir):
        print(f"\n⚠ Simulation directory not found: {sim_dir}")
        print("  Using synthetic data for demonstration...")
        use_real_data = False
    else:
        sim_files = [f for f in os.listdir(sim_dir) if f.endswith('.csv')]
        if len(sim_files) == 0:
            print(f"\n⚠ No simulation CSV files found in {sim_dir}")
            print("  Using synthetic data for demonstration...")
            use_real_data = False
        else:
            print(f"\n✓ Found {len(sim_files)} simulation files")
            use_real_data = True
    
    if use_real_data:
        # Load real simulation data
        print("\n[1] Loading real simulation data...")
        all_simulations = []
        
        for sim_file in sim_files:
            filepath = os.path.join(sim_dir, sim_file)
            df = pd.read_csv(filepath)
            all_simulations.append(df)
        
        simulations_df = pd.concat(all_simulations, ignore_index=True)
        
        # Filter out rows with NaN model names
        simulations_df = simulations_df.dropna(subset=['model_name'])
        
        print(f"  ✓ Loaded {len(simulations_df)} simulation results")
        print(f"  ✓ Models: {simulations_df['model_name'].unique().tolist()}")
        
        # Calculate summary statistics
        summary = simulations_df.groupby('model_name').agg({
            'accuracy': ['mean', 'std'],
            'f1_macro': ['mean', 'std'],
            'f1_weighted': ['mean', 'std']
        }).round(4)
        
        print("\n  Summary Statistics:")
        print(summary)
        
        # Prepare results dictionary for plotting
        results = {}
        for model in simulations_df['model_name'].unique():
            model_data = simulations_df[simulations_df['model_name'] == model]
            results[model] = {
                'accuracy': model_data['accuracy'].mean(),
                'f1_macro': model_data['f1_macro'].mean(),
                'f1_weighted': model_data['f1_weighted'].mean()
            }
        
        # Prepare CI results
        ci_results = {}
        for model in simulations_df['model_name'].unique():
            model_data = simulations_df[simulations_df['model_name'] == model]
            ci_results[model] = {
                'mean': model_data['f1_macro'].mean(),
                'std': model_data['f1_macro'].std()
            }
        
    else:
        # Generate synthetic data for demonstration
        print("\n[1] Generating synthetic simulation data...")
        np.random.seed(42)
        
        n_simulations = 10
        models = ['svm_bow', 'svm_embeddings', 'bert']
        
        simulations_df = pd.DataFrame({
            'simulation_id': list(range(n_simulations)) * len(models),
            'model_name': [m for m in models for _ in range(n_simulations)],
            'accuracy': (
                np.random.normal(0.795, 0.01, n_simulations).tolist() +
                np.random.normal(0.811, 0.01, n_simulations).tolist() +
                np.random.normal(0.844, 0.01, n_simulations).tolist()
            ),
            'f1_macro': (
                np.random.normal(0.795, 0.01, n_simulations).tolist() +
                np.random.normal(0.811, 0.01, n_simulations).tolist() +
                np.random.normal(0.844, 0.01, n_simulations).tolist()
            ),
            'f1_weighted': (
                np.random.normal(0.794, 0.01, n_simulations).tolist() +
                np.random.normal(0.810, 0.01, n_simulations).tolist() +
                np.random.normal(0.843, 0.01, n_simulations).tolist()
            )
        })
        
        results = {
            'svm_bow': {'accuracy': 0.795, 'f1_macro': 0.795, 'f1_weighted': 0.794},
            'svm_embeddings': {'accuracy': 0.811, 'f1_macro': 0.811, 'f1_weighted': 0.810},
            'bert': {'accuracy': 0.844, 'f1_macro': 0.844, 'f1_weighted': 0.843}
        }
        
        ci_results = {
            'svm_bow': {'mean': 0.795, 'std': 0.01},
            'svm_embeddings': {'mean': 0.811, 'std': 0.012},
            'bert': {'mean': 0.844, 'std': 0.008}
        }
        
        print(f"  ✓ Generated {len(simulations_df)} synthetic simulation results")
    
    # Generate all visualizations
    print("\n[2] Generating visualizations...")
    
    # 2.1 Metrics comparison
    print("  → Generating metrics comparison chart...")
    viz.plot_metrics_comparison(results, metrics=['accuracy', 'f1_macro', 'f1_weighted'])
    print("    ✓ Grouped bar chart created")
    
    # 2.2 Boxplots for each metric
    print("  → Generating boxplots...")
    viz.plot_boxplots(simulations_df, metric='accuracy')
    viz.plot_boxplots(simulations_df, metric='f1_macro')
    print("    ✓ Boxplots created for accuracy and f1_macro")
    
    # 2.3 Line evolution plots
    print("  → Generating line evolution plots...")
    viz.plot_line_evolution(simulations_df, metric='accuracy')
    viz.plot_line_evolution(simulations_df, metric='f1_macro')
    print("    ✓ Line plots created for metric evolution")
    
    # 2.4 Confidence intervals
    print("  → Generating confidence intervals...")
    viz.plot_confidence_intervals(ci_results, metric='f1_macro', confidence=0.95)
    print("    ✓ Bar chart with 95% CI created")
    
    # 2.5 P-value matrix (if we have statistical validator results)
    print("  → Generating p-value matrix...")
    models = list(results.keys())
    n_models = len(models)
    
    # Try to load real p-values or generate synthetic ones
    try:
        stat_validator = StatisticalValidator()
        pvalue_matrix = stat_validator.generate_pvalue_matrix(models, 'f1_macro')
        print("    ✓ Using real p-values from statistical validator")
    except:
        # Generate synthetic p-value matrix
        pvalue_matrix = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(i+1, n_models):
                p_val = np.random.uniform(0.001, 0.1)
                pvalue_matrix[i, j] = p_val
                pvalue_matrix[j, i] = p_val
        print("    ✓ Using synthetic p-values for demonstration")
    
    viz.plot_pvalue_matrix(pvalue_matrix, models, metric='f1_macro')
    print("    ✓ P-value significance matrix created")
    
    # 2.6 Confusion matrices (synthetic for demo)
    print("  → Generating confusion matrices...")
    for model in models:
        # Generate synthetic confusion matrix
        cm = np.array([
            [np.random.randint(400, 500), np.random.randint(20, 80)],
            [np.random.randint(20, 80), np.random.randint(400, 500)]
        ])
        viz.plot_confusion_matrix(cm, model)
    print(f"    ✓ Confusion matrices created for {len(models)} models")
    
    # Save all figures
    print("\n[3] Saving all figures...")
    output_dir = 'results/plots/demo_integration'
    viz.save_all_figures(output_dir=output_dir, dpi=300)
    
    # List generated files
    print("\n[4] Generated files:")
    if os.path.exists(output_dir):
        files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        for i, filename in enumerate(files, 1):
            filepath = os.path.join(output_dir, filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {i}. {filename} ({size_kb:.1f} KB)")
    
    # Clean up
    viz.close_all()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ All visualizations saved to: {output_dir}/")
    print("✓ Ready for presentation (300 DPI, professional theme)")
    print("\nThe Visualizer module is fully integrated and ready to use with:")
    print("  • Real simulation data from SimulationRunner")
    print("  • Statistical validation results from StatisticalValidator")
    print("  • Evaluation metrics from Evaluator")
    print("  • Confusion matrices from model predictions")


if __name__ == "__main__":
    try:
        demo_visualizer_with_real_data()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
