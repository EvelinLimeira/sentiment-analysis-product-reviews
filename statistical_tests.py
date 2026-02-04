# src/statistical_tests.py

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalValidator:
    """Validação estatística para comparação de modelos"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def load_simulations(self, model_name):
        """Carrega resultados de múltiplas simulações"""
        df = pd.read_csv(f'results/simulations/{model_name}_simulations.csv')
        return df
    
    def wilcoxon_pairwise(self, model1_name, model2_name, metric='f1_score'):
        """
        Teste de Wilcoxon pareado entre dois modelos
        
        Returns:
            dict: {'statistic': float, 'p_value': float, 
                   'significant': bool, 'winner': str}
        """
        # Carrega dados
        model1_data = self.load_simulations(model1_name)[metric].values
        model2_data = self.load_simulations(model2_name)[metric].values
        
        # Teste de Wilcoxon
        statistic, p_value = stats.wilcoxon(model1_data, model2_data, 
                                            alternative='two-sided')
        
        # Determina vencedor
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
        Teste de Kruskal-Wallis para múltiplos modelos
        
        Returns:
            dict: {'statistic': float, 'p_value': float, 'significant': bool}
        """
        # Carrega dados de todos os modelos
        data_groups = [self.load_simulations(name)[metric].values 
                      for name in model_names]
        
        # Teste de Kruskal-Wallis
        statistic, p_value = stats.kruskal(*data_groups)
        
        return {
            'models': model_names,
            'metric': metric,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
    
    def shapiro_normality(self, model_name, metric='f1_score'):
        """Teste de normalidade de Shapiro-Wilk"""
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
        """Gera matriz de p-valores para todos os pares"""
        n = len(model_names)
        matrix = np.zeros((n, n))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    result = self.wilcoxon_pairwise(model1, model2, metric)
                    matrix[i, j] = result['p_value']
                else:
                    matrix[i, j] = 1.0  # Diagonal
        
        # Visualização
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.4f', 
                   xticklabels=model_names, yticklabels=model_names,
                   cmap='RdYlGn_r', vmin=0, vmax=0.1,
                   cbar_kws={'label': 'p-value'})
        plt.title(f'Matriz de p-valores - {metric}\n(Verde: p<0.05 = significativo)')
        plt.tight_layout()
        plt.savefig(f'results/plots/statistical/pvalue_matrix_{metric}.png', dpi=300)
        plt.close()
        
        return matrix
    
    def generate_report(self, model_names, metrics=['accuracy', 'f1_score']):
        """Gera relatório completo de análise estatística"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("RELATÓRIO DE VALIDAÇÃO ESTATÍSTICA")
        report_lines.append("="*80)
        report_lines.append(f"\nNível de significância: α = {self.alpha} (95% de confiança)")
        report_lines.append(f"Modelos analisados: {', '.join(model_names)}\n")
        
        # Teste de Kruskal-Wallis
        report_lines.append("\n" + "="*80)
        report_lines.append("TESTE DE KRUSKAL-WALLIS (múltiplos grupos)")
        report_lines.append("="*80)
        
        for metric in metrics:
            kw_result = self.kruskal_wallis_multiple(model_names, metric)
            report_lines.append(f"\nMétrica: {metric.upper()}")
            report_lines.append(f"  Estatística H: {kw_result['statistic']:.4f}")
            report_lines.append(f"  p-valor: {kw_result['p_value']:.6f}")
            report_lines.append(f"  Significativo: {'SIM' if kw_result['significant'] else 'NÃO'}")
            if kw_result['significant']:
                report_lines.append(f"  → HÁ diferença significativa entre os modelos")
            else:
                report_lines.append(f"  → NÃO há diferença significativa entre os modelos")
        
        # Testes de Wilcoxon pareados
        report_lines.append("\n" + "="*80)
        report_lines.append("TESTES DE WILCOXON PAREADOS")
        report_lines.append("="*80)
        
        for metric in metrics:
            report_lines.append(f"\n{'─'*80}")
            report_lines.append(f"Métrica: {metric.upper()}")
            report_lines.append(f"{'─'*80}")
            
            # Comparações par-a-par
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    result = self.wilcoxon_pairwise(model1, model2, metric)
                    report_lines.append(f"\n{model1} vs {model2}:")
                    report_lines.append(f"  Mediana {model1}: {result['model1_median']:.4f}")
                    report_lines.append(f"  Mediana {model2}: {result['model2_median']:.4f}")
                    report_lines.append(f"  p-valor: {result['p_value']:.6f}")
                    report_lines.append(f"  Significativo: {'SIM' if result['significant'] else 'NÃO'}")
                    report_lines.append(f"  Vencedor: {result['winner']}")
        
        # Salva relatório
        report_text = "\n".join(report_lines)
        with open('results/statistical_tests/statistical_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n✓ Relatório estatístico gerado: results/statistical_tests/statistical_report.txt")
        return report_text


# EXEMPLO DE USO
if __name__ == "__main__":
    validator = StatisticalValidator(alpha=0.05)
    
    model_names = ['svm_bow', 'svm_embeddings', 'bert', 'llm']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Gera relatório completo
    validator.generate_report(model_names, metrics)
    
    # Gera matriz de p-valores para cada métrica
    for metric in metrics:
        validator.generate_pvalue_matrix(model_names, metric)