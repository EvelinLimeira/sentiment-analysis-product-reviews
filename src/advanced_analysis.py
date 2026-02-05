"""
Advanced NLP analysis module for sentiment analysis NLP project.

This module provides the AdvancedNLPAnalysis class for performing advanced
NLP-specific analyses including:
- Text length vs accuracy analysis
- Typo robustness analysis
- Emoji impact analysis
- Sarcasm/irony analysis
- Formality/dialect analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Optional
import logging
import os
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import emoji library (optional dependency)
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    logger.warning("emoji library not available. Emoji analysis will be limited.")


class AdvancedNLPAnalysis:
    """
    Advanced NLP-specific analyses.
    
    This class performs various advanced analyses to understand model behavior
    in different scenarios such as varying text lengths, presence of typos,
    emojis, sarcasm, and formality levels.
    
    Attributes:
        test_df: DataFrame with test data including 'text' and 'label' columns
        predictions: Dictionary mapping model names to prediction arrays
        output_dir: Directory to save analysis results
    """
    
    def __init__(
        self, 
        test_df: pd.DataFrame, 
        predictions_dict: Dict[str, np.ndarray],
        output_dir: str = 'results/advanced_analysis'
    ):
        """
        Initialize the AdvancedNLPAnalysis.
        
        Args:
            test_df: DataFrame with ['text', 'label'] columns (minimum)
            predictions_dict: Dictionary mapping model names to prediction arrays
                             Format: {'model_name': np.array([predictions])}
            output_dir: Directory to save analysis results (default: 'results/advanced_analysis')
            
        Raises:
            ValueError: If test_df is missing required columns or predictions don't match
        """
        # Validate test_df
        required_cols = {'text', 'label'}
        if not required_cols.issubset(test_df.columns):
            raise ValueError(
                f"test_df must contain columns {required_cols}. "
                f"Found: {set(test_df.columns)}"
            )
        
        self.test_df = test_df.copy()
        self.predictions = predictions_dict
        
        # Validate predictions match test_df length
        for model_name, preds in predictions_dict.items():
            if len(preds) != len(test_df):
                raise ValueError(
                    f"Predictions for {model_name} have length {len(preds)}, "
                    f"but test_df has length {len(test_df)}"
                )
        
        # Create output directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path('results/plots/advanced_analysis')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Add text length if not present
        if 'length' not in self.test_df.columns:
            self.test_df['length'] = self.test_df['text'].str.len()
        
        logger.info(f"AdvancedNLPAnalysis initialized with {len(test_df)} test samples")
        logger.info(f"Models: {list(predictions_dict.keys())}")
        logger.info(f"Output directory: {self.output_dir}")
    
    # ═════════════════════════════════════════════════════════════════
    # ANALYSIS 1: TEXT LENGTH vs ACCURACY
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_length_vs_accuracy(self) -> pd.DataFrame:
        """
        Analyzes relationship between text length and accuracy.
        
        Bins texts by character length and calculates accuracy for each bin.
        Also computes Pearson and Spearman correlations between length and accuracy.
        
        Returns:
            DataFrame with accuracy by length bin for each model
            
        Validates: Requirements 8.1, 8.2
        """
        logger.info("Analyzing text length vs accuracy...")
        
        # Define length bins (Requirement 8.1)
        bins = [0, 50, 100, 200, 500, float('inf')]
        labels = ['0-50', '51-100', '101-200', '201-500', '500+']
        
        self.test_df['length_bin'] = pd.cut(
            self.test_df['length'], 
            bins=bins, 
            labels=labels,
            include_lowest=True
        )
        
        results = []
        
        # Calculate accuracy per bin for each model
        for model_name, preds in self.predictions.items():
            self.test_df[f'{model_name}_correct'] = (preds == self.test_df['label'].values).astype(int)
            
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
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        for model_name in self.predictions.keys():
            model_data = results_df[results_df['model'] == model_name]
            plt.plot(
                model_data['length_bin'], 
                model_data['accuracy'], 
                marker='o', 
                label=model_name.replace('_', ' ').upper(), 
                linewidth=2,
                markersize=8
            )

        plt.xlabel('Text Length (characters)', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Accuracy vs Text Length', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        plot_path = self.plots_dir / 'length_vs_accuracy.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_path}")
        
        # Calculate correlations (Requirement 8.2)
        logger.info("\nCorrelation: Length × Accuracy")
        logger.info("="*60)
        
        correlation_results = []
        
        for model_name, preds in self.predictions.items():
            correct = (preds == self.test_df['label'].values).astype(int)
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(self.test_df['length'], correct)
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(self.test_df['length'], correct)
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
            logger.info(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
            
            correlation_results.append({
                'model': model_name,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_rho': spearman_r,
                'spearman_p': spearman_p
            })
        
        # Save results
        csv_path = self.output_dir / 'length_vs_accuracy.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"\nSaved results: {csv_path}")
        
        # Save correlation results
        corr_path = self.output_dir / 'length_correlation.csv'
        pd.DataFrame(correlation_results).to_csv(corr_path, index=False)
        logger.info(f"Saved correlations: {corr_path}")
        
        return results_df
    
    # ═════════════════════════════════════════════════════════════════
    # ANALYSIS 2: TYPO ROBUSTNESS
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_typo_robustness(
        self, 
        perturbed_predictions_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compares performance on clean text vs text with typos.
        
        Args:
            perturbed_predictions_dict: Predictions on perturbed dataset
                                       Format: {'model_name': np.array([predictions])}
            
        Returns:
            DataFrame with clean vs perturbed accuracy comparison
            
        Validates: Requirements 8.3, 8.4
        """
        logger.info("Analyzing typo robustness...")
        
        results = []
        
        for model_name in self.predictions.keys():
            if model_name not in perturbed_predictions_dict:
                logger.warning(f"No perturbed predictions for {model_name}, skipping")
                continue
            
            # Accuracy on clean dataset
            clean_acc = (self.predictions[model_name] == self.test_df['label'].values).mean()
            
            # Accuracy on perturbed dataset
            perturbed_acc = (perturbed_predictions_dict[model_name] == self.test_df['label'].values).mean()
            
            # Degradation (Requirement 8.4)
            degradation = clean_acc - perturbed_acc
            degradation_pct = (degradation / clean_acc) * 100 if clean_acc > 0 else 0
            
            results.append({
                'model': model_name,
                'clean_accuracy': clean_acc,
                'perturbed_accuracy': perturbed_acc,
                'degradation': degradation,
                'degradation_pct': degradation_pct
            })
            
            logger.info(
                f"{model_name}: Clean={clean_acc:.4f}, "
                f"Perturbed={perturbed_acc:.4f}, "
                f"Degradation={degradation_pct:.2f}%"
            )
        
        results_df = pd.DataFrame(results)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        bars1 = ax.bar(
            x - width/2, 
            results_df['clean_accuracy'], 
            width, 
            label='Clean Text', 
            alpha=0.8,
            color='#2ecc71'
        )
        bars2 = ax.bar(
            x + width/2, 
            results_df['perturbed_accuracy'], 
            width, 
            label='Text with Typos', 
            alpha=0.8,
            color='#e74c3c'
        )
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Robustness to Typos', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').upper() for m in results_df['model']], rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        # Add degradation percentage on top
        for i, row in results_df.iterrows():
            ax.text(
                i, 
                row['clean_accuracy'] + 0.03, 
                f"-{row['degradation_pct']:.1f}%",
                ha='center', 
                fontsize=10, 
                color='red',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'typo_robustness.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_path}")
        
        # Save results
        csv_path = self.output_dir / 'typo_robustness.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved results: {csv_path}")
        
        return results_df

    # ═════════════════════════════════════════════════════════════════
    # ANALYSIS 3: EMOJI ANALYSIS
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_emoji_impact(self) -> pd.DataFrame:
        """
        Analyzes impact of emojis on accuracy.
        
        Compares accuracy on reviews with vs without emojis.
        
        Returns:
            DataFrame with emoji impact analysis
            
        Validates: Requirement 8.5
        """
        logger.info("Analyzing emoji impact...")
        
        # Detect emojis
        if EMOJI_AVAILABLE:
            self.test_df['has_emoji'] = self.test_df['text'].apply(
                lambda x: bool(emoji.emoji_count(str(x)))
            )
        else:
            # Fallback: simple emoji detection using Unicode ranges
            def has_emoji_simple(text):
                emoji_pattern = re.compile(
                    "["
                    "\U0001F600-\U0001F64F"  # emoticons
                    "\U0001F300-\U0001F5FF"  # symbols & pictographs
                    "\U0001F680-\U0001F6FF"  # transport & map symbols
                    "\U0001F1E0-\U0001F1FF"  # flags
                    "\U00002702-\U000027B0"
                    "\U000024C2-\U0001F251"
                    "]+", 
                    flags=re.UNICODE
                )
                return bool(emoji_pattern.search(str(text)))
            
            self.test_df['has_emoji'] = self.test_df['text'].apply(has_emoji_simple)
        
        results = []
        
        for model_name, preds in self.predictions.items():
            # With emojis
            with_emoji_mask = self.test_df['has_emoji']
            if with_emoji_mask.sum() > 0:
                acc_with = (preds[with_emoji_mask] == self.test_df.loc[with_emoji_mask, 'label'].values).mean()
                count_with = with_emoji_mask.sum()
            else:
                acc_with = np.nan
                count_with = 0
            
            # Without emojis
            without_emoji_mask = ~self.test_df['has_emoji']
            if without_emoji_mask.sum() > 0:
                acc_without = (preds[without_emoji_mask] == self.test_df.loc[without_emoji_mask, 'label'].values).mean()
                count_without = without_emoji_mask.sum()
            else:
                acc_without = np.nan
                count_without = 0
            
            difference = acc_with - acc_without if not np.isnan(acc_with) and not np.isnan(acc_without) else np.nan
            
            results.append({
                'model': model_name,
                'accuracy_with_emoji': acc_with,
                'accuracy_without_emoji': acc_without,
                'difference': difference,
                'count_with_emoji': count_with,
                'count_without_emoji': count_without
            })
            
            logger.info(
                f"{model_name}: With emoji={acc_with:.4f} (n={count_with}), "
                f"Without={acc_without:.4f} (n={count_without})"
            )
        
        results_df = pd.DataFrame(results)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        ax.bar(
            x - width/2, 
            results_df['accuracy_with_emoji'], 
            width,
            label='With Emojis', 
            alpha=0.8,
            color='#f39c12'
        )
        ax.bar(
            x + width/2, 
            results_df['accuracy_without_emoji'], 
            width,
            label='Without Emojis', 
            alpha=0.8,
            color='#3498db'
        )
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Impact of Emojis on Accuracy', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').upper() for m in results_df['model']], rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'emoji_impact.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_path}")
        
        # Save results
        csv_path = self.output_dir / 'emoji_analysis.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved results: {csv_path}")
        
        return results_df
    
    # ═════════════════════════════════════════════════════════════════
    # ANALYSIS 4: SARCASM/IRONY
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_sarcasm(self, sarcasm_indices: List[int]) -> pd.DataFrame:
        """
        Analyzes performance on sarcastic reviews.
        
        Args:
            sarcasm_indices: List of indices of sarcastic reviews (manually annotated)
            
        Returns:
            DataFrame with sarcasm analysis results
            
        Validates: Requirement 8.6
        """
        logger.info(f"Analyzing sarcasm/irony (n={len(sarcasm_indices)} samples)...")
        
        sarcasm_mask = self.test_df.index.isin(sarcasm_indices)
        
        if sarcasm_mask.sum() == 0:
            logger.warning("No sarcastic samples found in test set")
            return pd.DataFrame()
        
        results = []
        
        for model_name, preds in self.predictions.items():
            # Sarcastic
            acc_sarcasm = (preds[sarcasm_mask] == self.test_df.loc[sarcasm_mask, 'label'].values).mean()
            
            # Non-sarcastic
            acc_normal = (preds[~sarcasm_mask] == self.test_df.loc[~sarcasm_mask, 'label'].values).mean()
            
            # Degradation
            degradation = acc_normal - acc_sarcasm
            degradation_pct = (degradation / acc_normal) * 100 if acc_normal > 0 else 0
            
            results.append({
                'model': model_name,
                'accuracy_sarcasm': acc_sarcasm,
                'accuracy_normal': acc_normal,
                'degradation': degradation,
                'degradation_pct': degradation_pct,
                'count_sarcasm': sarcasm_mask.sum(),
                'count_normal': (~sarcasm_mask).sum()
            })
            
            logger.info(
                f"{model_name}: Sarcasm={acc_sarcasm:.4f}, "
                f"Normal={acc_normal:.4f}, "
                f"Degradation={degradation_pct:.2f}%"
            )
        
        results_df = pd.DataFrame(results)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df))
        width = 0.35
        
        ax.bar(
            x - width/2, 
            results_df['accuracy_normal'], 
            width,
            label='Normal Reviews', 
            alpha=0.8,
            color='#2ecc71'
        )
        ax.bar(
            x + width/2, 
            results_df['accuracy_sarcasm'], 
            width,
            label='Sarcastic Reviews', 
            alpha=0.8,
            color='#9b59b6'
        )
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Performance on Sarcastic vs Normal Reviews', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').upper() for m in results_df['model']], rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        # Add degradation percentage
        for i, row in results_df.iterrows():
            ax.text(
                i, 
                row['accuracy_normal'] + 0.03, 
                f"-{row['degradation_pct']:.1f}%",
                ha='center', 
                fontsize=10, 
                color='red',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'sarcasm_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_path}")
        
        # Save results
        csv_path = self.output_dir / 'sarcasm_performance.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved results: {csv_path}")
        
        return results_df

    # ═════════════════════════════════════════════════════════════════
    # ANALYSIS 5: FORMALITY/DIALECT
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_formality(self) -> pd.DataFrame:
        """
        Analyzes sensitivity to formality/dialect.
        
        Categorizes reviews by formality level (formal, informal, excited)
        and analyzes accuracy for each category.
        
        Returns:
            DataFrame with formality analysis results
            
        Validates: Requirement 8.7
        """
        logger.info("Analyzing formality/dialect sensitivity...")
        
        # Detect slang/informality (English examples)
        slang_words = [
            'awesome', 'cool', 'sucks', 'lol', 'omg', 'gonna', 'wanna', 'gotta',
            'yeah', 'nope', 'yep', 'nah', 'dunno', 'kinda', 'sorta', 'ain\'t'
        ]
        
        def detect_formality(text):
            """Detect formality level of text."""
            text_str = str(text)
            text_lower = text_str.lower()
            
            # Check for informal language (slang)
            if any(slang in text_lower for slang in slang_words):
                return 'informal'
            # Check for excited language (caps, multiple exclamation marks)
            elif text_str.isupper() or '!!!' in text_str or '!!!' in text_str:
                return 'excited'
            else:
                return 'formal'
        
        self.test_df['formality'] = self.test_df['text'].apply(detect_formality)
        
        results = []
        
        for model_name, preds in self.predictions.items():
            for formality_type in ['formal', 'informal', 'excited']:
                mask = self.test_df['formality'] == formality_type
                if mask.sum() > 0:
                    acc = (preds[mask] == self.test_df.loc[mask, 'label'].values).mean()
                    count = mask.sum()
                    
                    results.append({
                        'model': model_name,
                        'formality': formality_type,
                        'accuracy': acc,
                        'count': count
                    })
                    
                    logger.info(f"{model_name} - {formality_type}: {acc:.4f} (n={count})")
        
        results_df = pd.DataFrame(results)
        
        # Create pivot table for heatmap
        pivot = results_df.pivot(index='model', columns='formality', values='accuracy')
        
        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt='.3f', 
            cmap='YlOrRd', 
            cbar_kws={'label': 'Accuracy'},
            linewidths=0.5,
            linecolor='gray'
        )
        plt.title('Accuracy by Formality Level', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Formality Level', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        
        # Update y-axis labels
        yticklabels = [label.get_text().replace('_', ' ').upper() for label in plt.gca().get_yticklabels()]
        plt.gca().set_yticklabels(yticklabels, rotation=0)
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'formality_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_path}")
        
        # Save results
        csv_path = self.output_dir / 'formality_analysis.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved results: {csv_path}")
        
        return results_df
    
    def run_all_analyses(
        self,
        perturbed_predictions: Optional[Dict[str, np.ndarray]] = None,
        sarcasm_indices: Optional[List[int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all available analyses.
        
        Args:
            perturbed_predictions: Optional predictions on perturbed dataset
            sarcasm_indices: Optional list of sarcastic review indices
            
        Returns:
            Dictionary mapping analysis names to result DataFrames
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING ALL ADVANCED NLP ANALYSES")
        logger.info("="*80 + "\n")
        
        results = {}
        
        # Analysis 1: Length vs Accuracy
        try:
            results['length_vs_accuracy'] = self.analyze_length_vs_accuracy()
        except Exception as e:
            logger.error(f"Error in length vs accuracy analysis: {e}", exc_info=True)
        
        # Analysis 2: Typo Robustness (if perturbed predictions provided)
        if perturbed_predictions:
            try:
                results['typo_robustness'] = self.analyze_typo_robustness(perturbed_predictions)
            except Exception as e:
                logger.error(f"Error in typo robustness analysis: {e}", exc_info=True)
        else:
            logger.info("Skipping typo robustness analysis (no perturbed predictions provided)")
        
        # Analysis 3: Emoji Impact
        try:
            results['emoji_impact'] = self.analyze_emoji_impact()
        except Exception as e:
            logger.error(f"Error in emoji impact analysis: {e}", exc_info=True)
        
        # Analysis 4: Sarcasm (if indices provided)
        if sarcasm_indices:
            try:
                results['sarcasm'] = self.analyze_sarcasm(sarcasm_indices)
            except Exception as e:
                logger.error(f"Error in sarcasm analysis: {e}", exc_info=True)
        else:
            logger.info("Skipping sarcasm analysis (no sarcasm indices provided)")
        
        # Analysis 5: Formality
        try:
            results['formality'] = self.analyze_formality()
        except Exception as e:
            logger.error(f"Error in formality analysis: {e}", exc_info=True)
        
        logger.info("\n" + "="*80)
        logger.info("ALL ANALYSES COMPLETE!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Plots saved to: {self.plots_dir}")
        logger.info("="*80 + "\n")
        
        return results


# USAGE EXAMPLE
if __name__ == "__main__":
    """
    Example usage of AdvancedNLPAnalysis.
    
    This demonstrates how to use the analysis module with your trained models.
    """
    import sys
    sys.path.append('.')
    
    from src.data_loader import DataLoader
    from src.preprocessor import Preprocessor
    from src.vectorizers import BoWVectorizer
    from src.embedding_encoder import EmbeddingEncoder
    from src.classifiers import SVMClassifier
    from src.bert_classifier import BERTClassifier
    import joblib
    
    # Load test data
    print("Loading test data...")
    data_loader = DataLoader(random_state=42)
    train_df, val_df, test_df = data_loader.load()
    
    # Load trained models and make predictions
    print("\nLoading models and generating predictions...")
    predictions = {}
    
    # Example: Load SVM + BoW model
    try:
        model_dir = Path('results/models/svm_bow')
        preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
        vectorizer = joblib.load(model_dir / 'vectorizer.pkl')
        classifier = joblib.load(model_dir / 'classifier.pkl')
        
        # Preprocess and predict
        test_texts_processed = preprocessor.transform(test_df['text'].tolist())
        X_test = vectorizer.transform(test_texts_processed)
        predictions['svm_bow'] = classifier.predict(X_test)
        print("✓ Loaded SVM + BoW")
    except Exception as e:
        print(f"⚠ Could not load SVM + BoW: {e}")
    
    # Example: Load SVM + Embeddings model
    try:
        model_dir = Path('results/models/svm_embeddings')
        preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
        encoder = joblib.load(model_dir / 'encoder.pkl')
        classifier = joblib.load(model_dir / 'classifier.pkl')
        
        # Preprocess and predict
        test_texts_processed = preprocessor.transform(test_df['text'].tolist())
        X_test = encoder.encode_batch(test_texts_processed)
        predictions['svm_embeddings'] = classifier.predict(X_test)
        print("✓ Loaded SVM + Embeddings")
    except Exception as e:
        print(f"⚠ Could not load SVM + Embeddings: {e}")
    
    # Example: Load BERT model
    try:
        model_dir = Path('results/models/bert/bert_model')
        classifier = BERTClassifier.load_model(str(model_dir))
        
        # Predict
        predictions['bert'] = classifier.predict(test_df['text'].tolist())
        print("✓ Loaded BERT")
    except Exception as e:
        print(f"⚠ Could not load BERT: {e}")
    
    if not predictions:
        print("\n⚠ No models loaded. Please train models first.")
        sys.exit(1)
    
    # Create analyzer
    print("\nInitializing AdvancedNLPAnalysis...")
    analyzer = AdvancedNLPAnalysis(test_df, predictions)
    
    # Run all available analyses
    print("\nRunning analyses...")
    results = analyzer.run_all_analyses()
    
    print("\n✓ Analysis complete!")
    print(f"  Results: {analyzer.output_dir}")
    print(f"  Plots: {analyzer.plots_dir}")

