# src/advanced_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import emoji
import re

class AdvancedNLPAnalysis:
    """Análises avançadas específicas de NLP"""
    
    def __init__(self, test_df, predictions_dict):
        """
        Args:
            test_df: DataFrame com ['text', 'label', 'length', 'has_emoji', etc.]
            predictions_dict: {'model_name': [predictions]}
        """
        self.test_df = test_df
        self.predictions = predictions_dict
        
    # ═════════════════════════════════════════════════════════════════
    # ANÁLISE 1: COMPRIMENTO DE TEXTO vs ACURÁCIA
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_length_vs_accuracy(self):
        """Analisa relação entre comprimento do texto e acurácia"""
        
        # Definir bins de comprimento
        bins = [0, 50, 100, 200, 500, 10000]
        labels = ['0-50', '51-100', '101-200', '201-500', '500+']
        
        self.test_df['length_bin'] = pd.cut(
            self.test_df['length'], 
            bins=bins, 
            labels=labels
        )
        
        results = []
        
        for model_name, preds in self.predictions.items():
            self.test_df[f'{model_name}_correct'] = (preds == self.test_df['label']).astype(int)
            
            for bin_label in labels:
                bin_mask = self.test_df['length_bin'] == bin_label
                if bin_mask.sum() > 0:
                    acc = self.test_df.loc[bin_mask, f'{model_name}_correct'].mean()
                    count = bin_mask.sum()
                    
                    results.append({
                        'model': model_name,
                        'length_bin': bin_label,
                        'accuracy': acc,
                        'count': count
                    })
        
        results_df = pd.DataFrame(results)
        
        # Visualização
        plt.figure(figsize=(12, 6))
        
        for model_name in self.predictions.keys():
            model_data = results_df[results_df['model'] == model_name]
            plt.plot(model_data['length_bin'], model_data['accuracy'], 
                    marker='o', label=model_name, linewidth=2)
        
        plt.xlabel('Comprimento do Texto (caracteres)', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.title('Acurácia vs Comprimento do Texto', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots/advanced_analysis/length_vs_accuracy.png', dpi=300)
        plt.close()
        
        # Correlação
        print("\n" + "="*60)
        print("CORRELAÇÃO: Comprimento × Acurácia")
        print("="*60)
        
        for model_name, preds in self.predictions.items():
            correct = (preds == self.test_df['label']).astype(int)
            pearson_r, pearson_p = pearsonr(self.test_df['length'], correct)
            spearman_r, spearman_p = spearmanr(self.test_df['length'], correct)
            
            print(f"\n{model_name}:")
            print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
            print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
        
        results_df.to_csv('results/advanced_analysis/length_vs_accuracy.csv', index=False)
        return results_df
    
    # ═════════════════════════════════════════════════════════════════
    # ANÁLISE 2: ROBUSTEZ A TYPOS
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_typo_robustness(self, perturbed_predictions_dict):
        """
        Compara performance em texto limpo vs texto com typos
        
        Args:
            perturbed_predictions_dict: Predições no dataset perturbado
        """
        results = []
        
        for model_name in self.predictions.keys():
            # Acurácia no dataset limpo
            clean_acc = (self.predictions[model_name] == self.test_df['label']).mean()
            
            # Acurácia no dataset perturbado
            perturbed_acc = (perturbed_predictions_dict[model_name] == self.test_df['label']).mean()
            
            # Degradação
            degradation = clean_acc - perturbed_acc
            degradation_pct = (degradation / clean_acc) * 100
            
            results.append({
                'model': model_name,
                'clean_accuracy': clean_acc,
                'perturbed_accuracy': perturbed_acc,
                'degradation': degradation,
                'degradation_pct': degradation_pct
            })
        
        results_df = pd.DataFrame(results)
        
        # Visualização
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        ax.bar(x - width/2, results_df['clean_accuracy'], width, 
               label='Texto Limpo', alpha=0.8)
        ax.bar(x + width/2, results_df['perturbed_accuracy'], width, 
               label='Texto com Typos', alpha=0.8)
        
        ax.set_xlabel('Modelo', fontsize=12)
        ax.set_ylabel('Acurácia', fontsize=12)
        ax.set_title('Robustez a Erros de Digitação', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['model'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar % de degradação no topo
        for i, row in results_df.iterrows():
            ax.text(i, row['clean_accuracy'] + 0.02, 
                   f"-{row['degradation_pct']:.1f}%",
                   ha='center', fontsize=9, color='red')
        
        plt.tight_layout()
        plt.savefig('results/plots/advanced_analysis/typo_robustness.png', dpi=300)
        plt.close()
        
        print("\n" + "="*60)
        print("ROBUSTEZ A TYPOS")
        print("="*60)
        print(results_df.to_string(index=False))
        
        results_df.to_csv('results/advanced_analysis/typo_robustness.csv', index=False)
        return results_df
    
    # ═════════════════════════════════════════════════════════════════
    # ANÁLISE 3: ANÁLISE DE EMOJIS
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_emoji_impact(self):
        """Analisa impacto de emojis na acurácia"""
        
        # Detectar emojis
        self.test_df['has_emoji'] = self.test_df['text'].apply(
            lambda x: bool(emoji.emoji_count(x))
        )
        
        results = []
        
        for model_name, preds in self.predictions.items():
            # Com emojis
            with_emoji_mask = self.test_df['has_emoji']
            if with_emoji_mask.sum() > 0:
                acc_with = (preds[with_emoji_mask] == self.test_df.loc[with_emoji_mask, 'label']).mean()
            else:
                acc_with = np.nan
            
            # Sem emojis
            without_emoji_mask = ~self.test_df['has_emoji']
            if without_emoji_mask.sum() > 0:
                acc_without = (preds[without_emoji_mask] == self.test_df.loc[without_emoji_mask, 'label']).mean()
            else:
                acc_without = np.nan
            
            results.append({
                'model': model_name,
                'accuracy_with_emoji': acc_with,
                'accuracy_without_emoji': acc_without,
                'difference': acc_with - acc_without,
                'count_with_emoji': with_emoji_mask.sum(),
                'count_without_emoji': without_emoji_mask.sum()
            })
        
        results_df = pd.DataFrame(results)
        
        # Visualização
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        ax.bar(x - width/2, results_df['accuracy_with_emoji'], width,
               label='Com Emojis', alpha=0.8)
        ax.bar(x + width/2, results_df['accuracy_without_emoji'], width,
               label='Sem Emojis', alpha=0.8)
        
        ax.set_xlabel('Modelo', fontsize=12)
        ax.set_ylabel('Acurácia', fontsize=12)
        ax.set_title('Impacto de Emojis na Acurácia', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['model'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/plots/advanced_analysis/emoji_impact.png', dpi=300)
        plt.close()
        
        print("\n" + "="*60)
        print("IMPACTO DE EMOJIS")
        print("="*60)
        print(results_df.to_string(index=False))
        
        results_df.to_csv('results/advanced_analysis/emoji_analysis.csv', index=False)
        return results_df
    
    # ═════════════════════════════════════════════════════════════════
    # ANÁLISE 4: SARCASMO/IRONIA
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_sarcasm(self, sarcasm_indices):
        """
        Analisa performance em reviews sarcásticos
        
        Args:
            sarcasm_indices: Lista de índices de reviews sarcásticos
        """
        sarcasm_mask = self.test_df.index.isin(sarcasm_indices)
        
        results = []
        
        for model_name, preds in self.predictions.items():
            # Sarcásticos
            acc_sarcasm = (preds[sarcasm_mask] == self.test_df.loc[sarcasm_mask, 'label']).mean()
            
            # Não-sarcásticos
            acc_normal = (preds[~sarcasm_mask] == self.test_df.loc[~sarcasm_mask, 'label']).mean()
            
            # Degradação
            degradation = acc_normal - acc_sarcasm
            degradation_pct = (degradation / acc_normal) * 100 if acc_normal > 0 else 0
            
            results.append({
                'model': model_name,
                'accuracy_sarcasm': acc_sarcasm,
                'accuracy_normal': acc_normal,
                'degradation': degradation,
                'degradation_pct': degradation_pct,
                'count_sarcasm': sarcasm_mask.sum()
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("ANÁLISE DE SARCASMO/IRONIA")
        print("="*60)
        print(results_df.to_string(index=False))
        
        results_df.to_csv('results/advanced_analysis/sarcasm_performance.csv', index=False)
        return results_df
    
    # ═════════════════════════════════════════════════════════════════
    # ANÁLISE 5: FORMALIDADE/DIALETO
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_formality(self):
        """Analisa sensibilidade a formalidade/dialeto"""
        
        # Detectar gírias/informalidade
        girias = ['top', 'massa', 'show', 'da hora', 'mt', 'vc', 'tbm', 'blz']
        
        def detect_formality(text):
            text_lower = text.lower()
            if any(giria in text_lower for giria in girias):
                return 'informal'
            elif text.isupper() or '!!!' in text:
                return 'excited'
            else:
                return 'formal'
        
        self.test_df['formality'] = self.test_df['text'].apply(detect_formality)
        
        results = []
        
        for model_name, preds in self.predictions.items():
            for formality_type in ['formal', 'informal', 'excited']:
                mask = self.test_df['formality'] == formality_type
                if mask.sum() > 0:
                    acc = (preds[mask] == self.test_df.loc[mask, 'label']).mean()
                    count = mask.sum()
                    
                    results.append({
                        'model': model_name,
                        'formality': formality_type,
                        'accuracy': acc,
                        'count': count
                    })
        
        results_df = pd.DataFrame(results)
        
        # Heatmap
        pivot = results_df.pivot(index='model', columns='formality', values='accuracy')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Acurácia'})
        plt.title('Acurácia por Nível de Formalidade', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/plots/advanced_analysis/formality_heatmap.png', dpi=300)
        plt.close()
        
        print("\n" + "="*60)
        print("ANÁLISE DE FORMALIDADE")
        print("="*60)
        print(pivot.to_string())
        
        results_df.to_csv('results/advanced_analysis/formality_analysis.csv', index=False)
        return results_df


# EXEMPLO DE USO
if __name__ == "__main__":
    # Carregar dados de teste e predições
    test_df = pd.read_csv('data/processed/test.csv')
    
    predictions = {
        'svm_bow': np.load('results/predictions/svm_bow_test.npy'),
        'svm_embeddings': np.load('results/predictions/svm_emb_test.npy'),
        'bert': np.load('results/predictions/bert_test.npy'),
        'llm': np.load('results/predictions/llm_test.npy')
    }
    
    # Criar analisador
    analyzer = AdvancedNLPAnalysis(test_df, predictions)
    
    # Executar todas as análises
    analyzer.analyze_length_vs_accuracy()
    analyzer.analyze_emoji_impact()
    analyzer.analyze_formality()
    
    # Para análises que precisam de dados adicionais:
    # perturbed_preds = {...}  # Predições em dataset perturbado
    # analyzer.analyze_typo_robustness(perturbed_preds)
    
    # sarcasm_indices = [10, 25, 47, ...]  # Índices anotados manualmente
    # analyzer.analyze_sarcasm(sarcasm_indices)