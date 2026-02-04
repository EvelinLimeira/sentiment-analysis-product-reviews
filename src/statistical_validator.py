# src/statistical_validator.py

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalValidator:
    """Statistical validation for model comparison."""
    
    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        self.alpha = alpha
        
    def load_simulations(self, model_name):
        """
        Loads results from multiple simulations.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with simulation results
        """
        df = pd.read_csv(f'results/simulations/{model_name}_simulations.csv')
        return df
    
    def wilcoxon_pairwise(self, model1_name, model2_name, metric='f1_score'):
        """
        Wilcoxon Signed-Rank paired test between two models.
        
        Args:
            model1_name: First model name
            model2_name: Second model name
            metric: Metric to compare (default 'f1_score')
        
        Returns:
            dict: {'statistic': float, 'p_value': float, 
                   'significant': bool, 'winner': str}
        """
        # Load data
        model1_data = self.load_simulations(model1_name)[metric].values
        model2_data = self.load_simulations(model2_name)[metric].values
        
        # Wilcoxon test
        statistic, p_value = stats.wilcoxon(model1_data, model2_data, 
                                            alternative='two-sided')
        
        # Determine winner
        if p_value < self.alpha:
            winner = model1_name if np.median(model1_data) > np.median(model2_data) else model2_name
            significant = True
        else:
            winner = "No significant difference"
            significant = False
        
        return {
            'model1': model1_name,
            'model2': model2_name,
            'metric': metric,
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'winner': winner,
            'model1_median': np.median(model1_data),
            'model2_median': np.median(model2_data)
        }

    def kruskal_wallis_multiple(self, model_names, metric='f1_score'):
        """
        Kruskal-Wallis H-test for multiple models.
        
        Args:
            model_names: List of model names
            metric: Metric to compare (default 'f1_score')
        
        Returns:
            dict: {'statistic': float, 'p_value': float, 'significant': bool}
        """
        # Load data from all models
        data_groups = [self.load_simulations(name)[metric].values 
                      for name in model_names]
        
        # Kruskal-Wallis test
        statistic, p_value = stats.kruskal(*data_groups)
        
        return {
            'models': model_names,
            'metric': metric,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def shapiro_normality(self, model_name, metric='f1_score'):
        """
        Shapiro-Wilk normality test.
        
        Args:
            model_name: Model name
            metric: Metric to test (default 'f1_score')
            
        Returns:
            dict with test results
        """
        data = self.load_simulations(model_name)[metric].values
        statistic, p_value = stats.shapiro(data)
        
        return {
            'model': model_name,
            'metric': metric,
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > self.alpha
        }
    
    def generate_pvalue_matrix(self, model_names, metric='f1_score'):
        """
        Generates p-value matrix for all model pairs.
        
        Args:
            model_names: List of model names
            metric: Metric to compare
            
        Returns:
            numpy array with p-values
        """
        n = len(model_names)
        matrix = np.zeros((n, n))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    result = self.wilcoxon_pairwise(model1, model2, metric)
                    matrix[i, j] = result['p_value']
                else:
                    matrix[i, j] = 1.0  # Diagonal
        
        # Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.4f', 
                   xticklabels=model_names, yticklabels=model_names,
                   cmap='RdYlGn_r', vmin=0, vmax=0.1,
                   cbar_kws={'label': 'p-value'})
        plt.title(f'P-value Matrix - {metric}\n(Green: p<0.05 = significant)')
        plt.tight_layout()
        plt.savefig(f'results/plots/statistical/pvalue_matrix_{metric}.png', dpi=300)
        plt.close()
        
        return matrix

    def generate_report(self, model_names, metrics=['accuracy', 'f1_score']):
        """
        Generates complete statistical analysis report.
        
        Args:
            model_names: List of model names
            metrics: List of metrics to analyze
            
        Returns:
            str: Complete report text
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("STATISTICAL VALIDATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nSignificance level: α = {self.alpha} (95% confidence)")
        report_lines.append(f"Models analyzed: {', '.join(model_names)}\n")
        
        # Kruskal-Wallis test
        report_lines.append("\n" + "="*80)
        report_lines.append("KRUSKAL-WALLIS TEST (multiple groups)")
        report_lines.append("="*80)
        
        for metric in metrics:
            kw_result = self.kruskal_wallis_multiple(model_names, metric)
            report_lines.append(f"\nMetric: {metric.upper()}")
            report_lines.append(f"  H statistic: {kw_result['statistic']:.4f}")
            report_lines.append(f"  p-value: {kw_result['p_value']:.6f}")
            report_lines.append(f"  Significant: {'YES' if kw_result['significant'] else 'NO'}")
            if kw_result['significant']:
                report_lines.append(f"  → There IS significant difference between models")
            else:
                report_lines.append(f"  → There is NO significant difference between models")
        
        # Wilcoxon paired tests
        report_lines.append("\n" + "="*80)
        report_lines.append("WILCOXON PAIRED TESTS")
        report_lines.append("="*80)
        
        for metric in metrics:
            report_lines.append(f"\n{'─'*80}")
            report_lines.append(f"Metric: {metric.upper()}")
            report_lines.append(f"{'─'*80}")
            
            # Pairwise comparisons
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    result = self.wilcoxon_pairwise(model1, model2, metric)
                    report_lines.append(f"\n{model1} vs {model2}:")
                    report_lines.append(f"  Median {model1}: {result['model1_median']:.4f}")
                    report_lines.append(f"  Median {model2}: {result['model2_median']:.4f}")
                    report_lines.append(f"  p-value: {result['p_value']:.6f}")
                    report_lines.append(f"  Significant: {'YES' if result['significant'] else 'NO'}")
                    report_lines.append(f"  Winner: {result['winner']}")
        
        # Save report
        report_text = "\n".join(report_lines)
        with open('results/statistical_tests/statistical_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n✓ Statistical report generated: results/statistical_tests/statistical_report.txt")
        return report_text


# USAGE EXAMPLE
if __name__ == "__main__":
    validator = StatisticalValidator(alpha=0.05)
    
    model_names = ['svm_bow', 'svm_embeddings', 'bert', 'llm']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Generate complete report
    validator.generate_report(model_names, metrics)
    
    # Generate p-value matrix for each metric
    for metric in metrics:
        validator.generate_pvalue_matrix(model_names, metric)
