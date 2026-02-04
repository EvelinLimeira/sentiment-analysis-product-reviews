"""
Visualizer module for generating professional visualizations.

This module provides the Visualizer class for creating high-quality plots
for presentation, including metrics comparison, confusion matrices, boxplots,
line plots, p-value matrices, and confidence intervals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


class Visualizer:
    """Generates professional visualizations for presentation."""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the visualizer with professional styling.
        
        Args:
            style: Matplotlib style to use (default: seaborn-v0_8-whitegrid)
            figsize: Default figure size (default: (10, 6))
        """
        self.figsize = figsize
        self.figures: Dict[str, plt.Figure] = {}
        
        # Set professional style
        try:
            plt.style.use(style)
        except:
            # Fallback to seaborn default if specific style not available
            sns.set_theme(style='whitegrid')
        
        # Set seaborn color palette
        self.palette = sns.color_palette('Set2')
        sns.set_palette(self.palette)
        
        # Set default font sizes
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_metrics_comparison(
        self, 
        results: Dict[str, Dict[str, float]], 
        metrics: List[str] = ['accuracy', 'f1_macro']
    ) -> plt.Figure:
        """
        Generates grouped bar chart comparing metrics across models.
        
        Args:
            results: Dictionary mapping model names to metric dictionaries
            metrics: List of metrics to compare (default: ['accuracy', 'f1_macro'])
            
        Returns:
            Matplotlib figure object
        """
        # Prepare data
        models = list(results.keys())
        n_models = len(models)
        n_metrics = len(metrics)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set bar width and positions
        bar_width = 0.8 / n_metrics
        x = np.arange(n_models)
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            offset = (i - n_metrics/2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=metric.replace('_', ' ').title())
        
        # Customize plot
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').upper() for m in models], rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.figures['metrics_comparison'] = fig
        return fig
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        model_name: str,
        labels: List[str] = ['Negative', 'Positive']
    ) -> plt.Figure:
        """
        Generates heatmap confusion matrix.
        
        Args:
            cm: Confusion matrix as numpy array
            model_name: Name of the model
            labels: Class labels (default: ['Negative', 'Positive'])
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        # Customize plot
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(f'Confusion Matrix - {model_name.replace("_", " ").upper()}', 
                    fontweight='bold', pad=20)
        
        plt.tight_layout()
        fig_name = f'confusion_matrix_{model_name}'
        self.figures[fig_name] = fig
        return fig
    
    def plot_boxplots(
        self, 
        simulations_df: pd.DataFrame, 
        metric: str = 'f1_macro'
    ) -> plt.Figure:
        """
        Generates boxplots showing metric distribution across simulations.
        
        Args:
            simulations_df: DataFrame with columns ['model_name', metric]
            metric: Metric to plot (default: 'f1_macro')
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create boxplot
        models = simulations_df['model_name'].unique()
        data = [simulations_df[simulations_df['model_name'] == model][metric].values 
                for model in models]
        
        bp = ax.boxplot(
            data, 
            tick_labels=[m.replace('_', ' ').upper() for m in models],
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], self.palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize plot
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution Across Simulations', 
                    fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        fig_name = f'boxplot_{metric}'
        self.figures[fig_name] = fig
        return fig
    
    def plot_line_evolution(
        self, 
        simulations_df: pd.DataFrame, 
        metric: str = 'f1_macro'
    ) -> plt.Figure:
        """
        Generates line plot showing metric evolution across simulations.
        
        Args:
            simulations_df: DataFrame with columns ['simulation_id', 'model_name', metric]
            metric: Metric to plot (default: 'f1_macro')
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot line for each model
        models = simulations_df['model_name'].unique()
        for i, model in enumerate(models):
            model_data = simulations_df[simulations_df['model_name'] == model]
            ax.plot(
                model_data['simulation_id'], 
                model_data[metric],
                marker='o',
                label=model.replace('_', ' ').upper(),
                color=self.palette[i % len(self.palette)],
                linewidth=2,
                markersize=6
            )
        
        # Customize plot
        ax.set_xlabel('Simulation ID', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Evolution Across Simulations', 
                    fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_name = f'line_evolution_{metric}'
        self.figures[fig_name] = fig
        return fig
    
    def plot_pvalue_matrix(
        self, 
        matrix: np.ndarray, 
        model_names: List[str],
        metric: str = 'f1_macro'
    ) -> plt.Figure:
        """
        Generates p-value significance matrix with color coding.
        
        Green indicates p < 0.05 (significant difference).
        Red indicates p >= 0.05 (no significant difference).
        
        Args:
            matrix: P-value matrix as numpy array
            model_names: List of model names
            metric: Metric name for title (default: 'f1_macro')
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create custom colormap: green for p<0.05, red for p>=0.05
        # Using RdYlGn_r (reversed Red-Yellow-Green)
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.4f',
            xticklabels=[m.replace('_', ' ').upper() for m in model_names],
            yticklabels=[m.replace('_', ' ').upper() for m in model_names],
            cmap='RdYlGn_r',
            vmin=0,
            vmax=0.1,
            cbar_kws={'label': 'p-value'},
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )
        
        # Customize plot
        ax.set_title(
            f'P-value Matrix - {metric.replace("_", " ").title()}\n'
            f'(Green: p < 0.05 = significant difference)',
            fontweight='bold',
            pad=20
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        fig_name = f'pvalue_matrix_{metric}'
        self.figures[fig_name] = fig
        return fig
    
    def plot_confidence_intervals(
        self, 
        results: Dict[str, Dict[str, float]], 
        metric: str = 'f1_macro',
        confidence: float = 0.95
    ) -> plt.Figure:
        """
        Generates bar chart with 95% confidence intervals.
        
        Args:
            results: Dictionary with 'mean' and 'std' for each model
                    Format: {model_name: {'mean': float, 'std': float}}
            metric: Metric name for labels (default: 'f1_macro')
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data
        models = list(results.keys())
        means = [results[model]['mean'] for model in models]
        stds = [results[model]['std'] for model in models]
        
        # Calculate confidence intervals (assuming normal distribution)
        # For 95% CI: mean ± 1.96 * std
        from scipy import stats as scipy_stats
        z_score = scipy_stats.norm.ppf((1 + confidence) / 2)
        errors = [z_score * std for std in stds]
        
        # Create bar chart
        x = np.arange(len(models))
        bars = ax.bar(
            x, 
            means, 
            yerr=errors,
            capsize=5,
            alpha=0.7,
            color=self.palette[:len(models)],
            edgecolor='black',
            linewidth=1.5
        )
        
        # Add value labels on bars
        for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + err + 0.01,
                f'{mean:.3f}±{err:.3f}',
                ha='center', 
                va='bottom',
                fontsize=9
            )
        
        # Customize plot
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(
            f'{metric.replace("_", " ").title()} with {int(confidence*100)}% Confidence Intervals',
            fontweight='bold',
            pad=20
        )
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').upper() for m in models], rotation=15, ha='right')
        ax.set_ylim(0, min(1.0, max(means) + max(errors) + 0.1))
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig_name = f'confidence_intervals_{metric}'
        self.figures[fig_name] = fig
        return fig
    
    def save_all_figures(self, output_dir: str = 'results/plots', dpi: int = 300) -> None:
        """
        Saves all generated figures in high resolution.
        
        Args:
            output_dir: Directory to save figures (default: 'results/plots')
            dpi: Resolution in dots per inch (default: 300 for high quality)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each figure
        saved_count = 0
        for fig_name, fig in self.figures.items():
            filepath = os.path.join(output_dir, f'{fig_name}.png')
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
            saved_count += 1
        
        if saved_count > 0:
            print(f"\n✓ Successfully saved {saved_count} figures to {output_dir}/")
        else:
            print("⚠ No figures to save. Generate plots first.")
    
    def close_all(self) -> None:
        """Closes all figure windows to free memory."""
        plt.close('all')
        self.figures.clear()


# Example usage
if __name__ == "__main__":
    # Example: Create visualizer and generate sample plots
    viz = Visualizer()
    
    # Sample data for demonstration
    results = {
        'svm_bow': {'accuracy': 0.796, 'f1_macro': 0.795},
        'svm_embeddings': {'accuracy': 0.812, 'f1_macro': 0.811},
        'bert': {'accuracy': 0.845, 'f1_macro': 0.844}
    }
    
    # Generate metrics comparison
    viz.plot_metrics_comparison(results)
    
    # Sample confusion matrix
    cm = np.array([[450, 50], [30, 470]])
    viz.plot_confusion_matrix(cm, 'svm_bow')
    
    # Sample simulation data
    sim_data = pd.DataFrame({
        'simulation_id': list(range(10)) * 3,
        'model_name': ['svm_bow']*10 + ['svm_embeddings']*10 + ['bert']*10,
        'f1_macro': np.random.normal(0.795, 0.01, 10).tolist() + 
                   np.random.normal(0.811, 0.01, 10).tolist() +
                   np.random.normal(0.844, 0.01, 10).tolist()
    })
    
    viz.plot_boxplots(sim_data)
    viz.plot_line_evolution(sim_data)
    
    # Sample p-value matrix
    pvalue_matrix = np.array([
        [1.0, 0.02, 0.001],
        [0.02, 1.0, 0.03],
        [0.001, 0.03, 1.0]
    ])
    viz.plot_pvalue_matrix(pvalue_matrix, ['svm_bow', 'svm_embeddings', 'bert'])
    
    # Sample confidence intervals
    ci_results = {
        'svm_bow': {'mean': 0.795, 'std': 0.01},
        'svm_embeddings': {'mean': 0.811, 'std': 0.012},
        'bert': {'mean': 0.844, 'std': 0.008}
    }
    viz.plot_confidence_intervals(ci_results)
    
    # Save all figures
    viz.save_all_figures()
    
    print("\n✓ Visualizer module demonstration complete!")
