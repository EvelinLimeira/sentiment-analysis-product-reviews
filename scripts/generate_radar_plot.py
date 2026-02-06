"""
Generate radar plot comparing model performance across multiple metrics.

This script creates a radar (spider) plot to visualize the performance of
different sentiment analysis models across various metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

def parse_metric_value(value_str):
    """Parse metric value from 'mean ± std' format."""
    if isinstance(value_str, str) and '±' in value_str:
        mean_str = value_str.split('±')[0].strip()
        return float(mean_str)
    return float(value_str)

def create_radar_plot(data_file='results/summary_statistics.csv', 
                     output_file='results/plots/radar_comparison.png'):
    """
    Create radar plot comparing models across metrics.
    
    Args:
        data_file: Path to summary statistics CSV
        output_file: Path to save the radar plot
    """
    # Read data
    df = pd.read_csv(data_file)
    
    # Parse metrics (extract mean values)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    models_data = {}
    for _, row in df.iterrows():
        model_name = row['Model']
        values = [parse_metric_value(row[metric]) for metric in metrics_to_plot]
        models_data[model_name] = values
    
    # Number of variables
    categories = metrics_to_plot
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for each model
    colors = {
        'BERT': '#2E86AB',
        'SVM+BoW': '#A23B72',
        'SVM+Embeddings': '#F18F01'
    }
    
    # Plot data for each model
    for model_name, values in models_data.items():
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                color=colors.get(model_name, '#333333'))
        ax.fill(angles, values, alpha=0.15, color=colors.get(model_name, '#333333'))
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, weight='bold')
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    # Add title
    plt.title('Model Performance Comparison\nAcross Multiple Metrics', 
              size=16, weight='bold', pad=20)
    
    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Radar plot saved to: {output_file}")
    
    plt.close()

def create_advanced_radar_plot(output_file='results/plots/radar_advanced_analysis.png'):
    """
    Create radar plot for advanced analysis metrics.
    
    Shows robustness across different text characteristics.
    """
    # Advanced analysis data (from your results)
    categories = ['Normal Text', 'With Typos', 'Sarcasm', 'Formal', 'Informal', 'Excited']
    
    # Data from advanced analysis results
    models_data = {
        'BERT': [0.9041, 0.8993, 0.8800, 0.9035, 0.8804, 0.9412],
        'SVM+BoW': [0.8007, 0.7940, 0.5600, 0.7833, 0.8696, 0.9020],
        'SVM+Embeddings': [0.7710, 0.7593, 0.6000, 0.7620, 0.7826, 0.8235]
    }
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    colors = {
        'BERT': '#2E86AB',
        'SVM+BoW': '#A23B72',
        'SVM+Embeddings': '#F18F01'
    }
    
    for model_name, values in models_data.items():
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2.5, label=model_name,
                color=colors[model_name], markersize=8)
        ax.fill(angles, values_plot, alpha=0.15, color=colors[model_name])
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, weight='bold')
    
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12)
    
    plt.title('Model Robustness Across Text Characteristics\n(Advanced Analysis)', 
              size=16, weight='bold', pad=20)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Advanced radar plot saved to: {output_file}")
    
    plt.close()

def main():
    """Generate all radar plots."""
    print("Generating radar plots...")
    print("-" * 60)
    
    # Standard metrics radar plot
    create_radar_plot()
    
    # Advanced analysis radar plot
    create_advanced_radar_plot()
    
    print("-" * 60)
    print("✓ All radar plots generated successfully!")
    print("\nGenerated plots:")
    print("  1. results/plots/radar_comparison.png")
    print("  2. results/plots/radar_advanced_analysis.png")

if __name__ == '__main__':
    main()
