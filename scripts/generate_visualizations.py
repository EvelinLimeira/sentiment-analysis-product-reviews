"""
Generate visualizations for BERT simulation results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualizer import Visualizer

def main():
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Load simulation results
    print("\nLoading simulation results...")
    bert_df = pd.read_csv('results/simulations/bert_simulations.csv')
    svm_bow_df = pd.read_csv('results/simulations/svm_bow_simulations.csv')
    svm_emb_df = pd.read_csv('results/simulations/svm_embeddings_simulations.csv')
    
    print(f"✓ BERT: {len(bert_df)} simulations")
    print(f"✓ SVM+BoW: {len(svm_bow_df)} simulations")
    print(f"✓ SVM+Embeddings: {len(svm_emb_df)} simulations")
    
    # Prepare data
    bert_df['model'] = 'BERT'
    svm_bow_df['model'] = 'SVM+BoW'
    svm_emb_df['model'] = 'SVM+Embeddings'
    
    all_df = pd.concat([bert_df, svm_bow_df, svm_emb_df], ignore_index=True)
    
    # Create visualizer
    viz = Visualizer()
    
    # 1. Boxplots for Accuracy and F1-Score
    print("\n[1/5] Creating boxplots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy boxplot
    data_acc = [
        bert_df['accuracy'].values,
        svm_bow_df['accuracy'].values,
        svm_emb_df['accuracy'].values
    ]
    bp1 = axes[0].boxplot(data_acc, labels=['BERT', 'SVM+BoW', 'SVM+Emb'], patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['#2ecc71', '#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Distribution (30 simulations)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.7, 1.0])
    
    # F1-Score boxplot
    data_f1 = [
        bert_df['f1_score'].values,
        svm_bow_df['f1_score'].values,
        svm_emb_df['f1_score'].values
    ]
    bp2 = axes[1].boxplot(data_f1, labels=['BERT', 'SVM+BoW', 'SVM+Emb'], patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['#2ecc71', '#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('F1-Score (Macro)', fontsize=12)
    axes[1].set_title('F1-Score Distribution (30 simulations)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/plots/boxplots_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plots/boxplots_comparison.png")
    plt.close()
    
    # 2. Metrics comparison bar chart
    print("[2/5] Creating metrics comparison...")
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(metrics))
    width = 0.25
    
    bert_means = [bert_df[m].mean() for m in metrics]
    svm_bow_means = [svm_bow_df[m].mean() for m in metrics]
    svm_emb_means = [svm_emb_df[m].mean() for m in metrics]
    
    ax.bar([i - width for i in x], bert_means, width, label='BERT', color='#2ecc71', alpha=0.8)
    ax.bar(x, svm_bow_means, width, label='SVM+BoW', color='#3498db', alpha=0.8)
    ax.bar([i + width for i in x], svm_emb_means, width, label='SVM+Embeddings', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (Mean across 30 simulations)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/plots/metrics_comparison_bar.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plots/metrics_comparison_bar.png")
    plt.close()
    
    # 3. Confidence intervals
    print("[3/5] Creating confidence intervals plot...")
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['BERT', 'SVM+BoW', 'SVM+Emb']
    dfs = [bert_df, svm_bow_df, svm_emb_df]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (model, df, color) in enumerate(zip(models, dfs, colors)):
        values = df['f1_score'].values
        mean = values.mean()
        std_err = stats.sem(values)
        ci = std_err * stats.t.ppf(0.975, len(values) - 1)
        
        ax.errorbar(i, mean, yerr=ci, fmt='o', markersize=10, 
                   capsize=10, capthick=2, color=color, label=model)
        ax.text(i, mean + ci + 0.01, f'{mean:.4f}', ha='center', fontsize=10)
    
    ax.set_ylabel('F1-Score (Macro)', fontsize=12)
    ax.set_title('F1-Score with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/plots/confidence_intervals_f1.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plots/confidence_intervals_f1.png")
    plt.close()
    
    # 4. Evolution across simulations
    print("[4/5] Creating evolution plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(bert_df['simulation_id'], bert_df['f1_score'], 
           marker='o', label='BERT', color='#2ecc71', linewidth=2)
    ax.plot(svm_bow_df['simulation_id'], svm_bow_df['f1_score'], 
           marker='s', label='SVM+BoW', color='#3498db', linewidth=2)
    ax.plot(svm_emb_df['simulation_id'], svm_emb_df['f1_score'], 
           marker='^', label='SVM+Embeddings', color='#e74c3c', linewidth=2)
    
    ax.set_xlabel('Simulation ID', fontsize=12)
    ax.set_ylabel('F1-Score (Macro)', fontsize=12)
    ax.set_title('F1-Score Evolution Across Simulations', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/evolution_f1.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/plots/evolution_f1.png")
    plt.close()
    
    # 5. Summary table
    print("[5/5] Creating summary table...")
    summary_data = []
    
    for model, df in [('BERT', bert_df), ('SVM+BoW', svm_bow_df), ('SVM+Embeddings', svm_emb_df)]:
        summary_data.append({
            'Model': model,
            'Accuracy': f"{df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}",
            'Precision': f"{df['precision'].mean():.4f} ± {df['precision'].std():.4f}",
            'Recall': f"{df['recall'].mean():.4f} ± {df['recall'].std():.4f}",
            'F1-Score': f"{df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}",
            'Train Time (s)': f"{df['training_time'].mean():.2f}",
            'Inference Time (s)': f"{df['inference_time'].mean():.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/summary_statistics.csv', index=False)
    print("  ✓ Saved: results/summary_statistics.csv")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/plots/boxplots_comparison.png")
    print("  - results/plots/metrics_comparison_bar.png")
    print("  - results/plots/confidence_intervals_f1.png")
    print("  - results/plots/evolution_f1.png")
    print("  - results/summary_statistics.csv")
    print()

if __name__ == '__main__':
    main()
