"""
Generate comprehensive F1-Score validation report with visualizations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    print("=" * 80)
    print("F1-SCORE VALIDATION REPORT")
    print("=" * 80)
    print()
    
    # Load simulation results
    print("Loading simulation results...")
    bert_df = pd.read_csv('results/simulations/bert_simulations.csv')
    svm_bow_df = pd.read_csv('results/simulations/svm_bow_simulations.csv')
    svm_emb_df = pd.read_csv('results/simulations/svm_embeddings_simulations.csv')
    
    # Determine F1 column name
    f1_col = 'f1_macro' if 'f1_macro' in bert_df.columns else 'f1_score'
    
    print(f"✓ Loaded {len(bert_df)} BERT simulations")
    print(f"✓ Loaded {len(svm_bow_df)} SVM+BoW simulations")
    print(f"✓ Loaded {len(svm_emb_df)} SVM+Embeddings simulations")
    print()
    
    # 1. Descriptive Statistics
    print("=" * 80)
    print("1. DESCRIPTIVE STATISTICS (F1-SCORE)")
    print("=" * 80)
    print()
    
    models = {
        'BERT': bert_df[f1_col].values,
        'SVM+BoW': svm_bow_df[f1_col].values,
        'SVM+Embeddings': svm_emb_df[f1_col].values
    }
    
    for name, values in models.items():
        print(f"{name}:")
        print(f"  Mean:     {values.mean():.4f}")
        print(f"  Median:   {np.median(values):.4f}")
        print(f"  Std Dev:  {values.std():.4f}")
        print(f"  Min:      {values.min():.4f}")
        print(f"  Max:      {values.max():.4f}")
        print(f"  Range:    {values.max() - values.min():.4f}")
        print()
    
    # 2. Confidence Intervals
    print("=" * 80)
    print("2. 95% CONFIDENCE INTERVALS (F1-SCORE)")
    print("=" * 80)
    print()
    
    for name, values in models.items():
        mean = values.mean()
        std_err = stats.sem(values)
        ci = std_err * stats.t.ppf(0.975, len(values) - 1)
        print(f"{name}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  95% CI: [{mean - ci:.4f}, {mean + ci:.4f}]")
        print(f"  Margin of Error: ±{ci:.4f}")
        print()
    
    # 3. Statistical Tests
    print("=" * 80)
    print("3. STATISTICAL SIGNIFICANCE TESTS (F1-SCORE)")
    print("=" * 80)
    print()
    
    # Wilcoxon tests
    comparisons = [
        ('BERT', 'SVM+BoW'),
        ('BERT', 'SVM+Embeddings'),
        ('SVM+BoW', 'SVM+Embeddings')
    ]
    
    for model1, model2 in comparisons:
        values1 = models[model1]
        values2 = models[model2]
        
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(values1, values2)
        
        mean_diff = values1.mean() - values2.mean()
        winner = model1 if mean_diff > 0 else model2
        
        print(f"{model1} vs {model2}:")
        print(f"  Mean {model1}: {values1.mean():.4f}")
        print(f"  Mean {model2}: {values2.mean():.4f}")
        print(f"  Mean Difference: {abs(mean_diff):.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant (α=0.05): {'YES ✓' if p_value < 0.05 else 'NO ✗'}")
        if p_value < 0.05:
            print(f"  Winner: {winner}")
        print()
    
    # 4. Effect Size (Cohen's d)
    print("=" * 80)
    print("4. EFFECT SIZE (COHEN'S D)")
    print("=" * 80)
    print()
    
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (x.mean() - y.mean()) / np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / dof)
    
    for model1, model2 in comparisons:
        values1 = models[model1]
        values2 = models[model2]
        d = cohens_d(values1, values2)
        
        if abs(d) < 0.2:
            magnitude = "negligible"
        elif abs(d) < 0.5:
            magnitude = "small"
        elif abs(d) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        print(f"{model1} vs {model2}:")
        print(f"  Cohen's d: {d:.4f}")
        print(f"  Magnitude: {magnitude}")
        print()
    
    # 5. Visualizations
    print("=" * 80)
    print("5. GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Create output directory
    Path('results/plots/f1_validation').mkdir(parents=True, exist_ok=True)
    
    # 5.1 Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [models['BERT'], models['SVM+BoW'], models['SVM+Embeddings']]
    bp = ax.boxplot(data, labels=['BERT', 'SVM+BoW', 'SVM+Emb'], patch_artist=True)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score Distribution Across Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/f1_validation/f1_boxplot.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plots/f1_validation/f1_boxplot.png")
    plt.close()
    
    # 5.2 Violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for violin plot
    plot_data = []
    plot_labels = []
    for name, values in models.items():
        plot_data.extend(values)
        plot_labels.extend([name] * len(values))
    
    df_plot = pd.DataFrame({'F1-Score': plot_data, 'Model': plot_labels})
    
    sns.violinplot(data=df_plot, x='Model', y='F1-Score', palette=colors, ax=ax)
    ax.set_title('F1-Score Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/f1_validation/f1_violin.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plots/f1_validation/f1_violin.png")
    plt.close()
    
    # 5.3 Confidence intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = range(len(models))
    means = []
    cis = []
    
    for name, values in models.items():
        mean = values.mean()
        std_err = stats.sem(values)
        ci = std_err * stats.t.ppf(0.975, len(values) - 1)
        means.append(mean)
        cis.append(ci)
    
    ax.errorbar(x_pos, means, yerr=cis, fmt='o', markersize=10, 
               capsize=10, capthick=2, color='black')
    
    for i, (mean, ci, color) in enumerate(zip(means, cis, colors)):
        ax.scatter(i, mean, s=200, color=color, alpha=0.7, zorder=3)
        ax.text(i, mean + ci + 0.01, f'{mean:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models.keys())
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/f1_validation/f1_confidence_intervals.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plots/f1_validation/f1_confidence_intervals.png")
    plt.close()
    
    # 6. Save detailed report
    print()
    print("=" * 80)
    print("6. SAVING DETAILED REPORT")
    print("=" * 80)
    print()
    
    report_path = 'results/f1_validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("F1-SCORE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        for name, values in models.items():
            mean = values.mean()
            std = values.std()
            std_err = stats.sem(values)
            ci = std_err * stats.t.ppf(0.975, len(values) - 1)
            f.write(f"\n{name}:\n")
            f.write(f"  F1-Score: {mean:.4f} ± {std:.4f}\n")
            f.write(f"  95% CI: [{mean - ci:.4f}, {mean + ci:.4f}]\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE\n")
        f.write("=" * 80 + "\n\n")
        
        for model1, model2 in comparisons:
            values1 = models[model1]
            values2 = models[model2]
            statistic, p_value = stats.wilcoxon(values1, values2)
            mean_diff = values1.mean() - values2.mean()
            winner = model1 if mean_diff > 0 else model2
            
            f.write(f"{model1} vs {model2}:\n")
            f.write(f"  p-value: {p_value:.6f}\n")
            f.write(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}\n")
            if p_value < 0.05:
                f.write(f"  Winner: {winner}\n")
            f.write("\n")
    
    print(f"  ✓ Saved: {report_path}")
    
    print()
    print("=" * 80)
    print("F1-SCORE VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - results/plots/f1_validation/f1_boxplot.png")
    print("  - results/plots/f1_validation/f1_violin.png")
    print("  - results/plots/f1_validation/f1_confidence_intervals.png")
    print("  - results/f1_validation_report.txt")
    print()

if __name__ == '__main__':
    main()
