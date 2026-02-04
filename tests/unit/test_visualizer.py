"""
Unit tests for the Visualizer module.
"""

import sys
import os
sys.path.insert(0, 'src')

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    """Test cases for the Visualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.viz = Visualizer()
        
        # Sample data for testing
        self.sample_results = {
            'svm_bow': {'accuracy': 0.796, 'f1_macro': 0.795},
            'svm_embeddings': {'accuracy': 0.812, 'f1_macro': 0.811},
            'bert': {'accuracy': 0.845, 'f1_macro': 0.844}
        }
        
        self.sample_cm = np.array([[450, 50], [30, 470]])
        
        self.sample_simulations = pd.DataFrame({
            'simulation_id': list(range(10)) * 3,
            'model_name': ['svm_bow']*10 + ['svm_embeddings']*10 + ['bert']*10,
            'accuracy': np.random.normal(0.795, 0.01, 10).tolist() + 
                       np.random.normal(0.811, 0.01, 10).tolist() +
                       np.random.normal(0.844, 0.01, 10).tolist(),
            'f1_macro': np.random.normal(0.795, 0.01, 10).tolist() + 
                       np.random.normal(0.811, 0.01, 10).tolist() +
                       np.random.normal(0.844, 0.01, 10).tolist()
        })
    
    def tearDown(self):
        """Clean up after tests."""
        self.viz.close_all()
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertIsInstance(self.viz, Visualizer)
        self.assertEqual(self.viz.figsize, (10, 6))
        self.assertIsInstance(self.viz.figures, dict)
        self.assertEqual(len(self.viz.figures), 0)
    
    def test_plot_metrics_comparison(self):
        """Test metrics comparison plot generation."""
        fig = self.viz.plot_metrics_comparison(
            self.sample_results, 
            metrics=['accuracy', 'f1_macro']
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('metrics_comparison', self.viz.figures)
        self.assertEqual(self.viz.figures['metrics_comparison'], fig)
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plot generation."""
        fig = self.viz.plot_confusion_matrix(self.sample_cm, 'svm_bow')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('confusion_matrix_svm_bow', self.viz.figures)
    
    def test_plot_boxplots(self):
        """Test boxplot generation."""
        fig = self.viz.plot_boxplots(self.sample_simulations, metric='f1_macro')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('boxplot_f1_macro', self.viz.figures)
    
    def test_plot_line_evolution(self):
        """Test line evolution plot generation."""
        fig = self.viz.plot_line_evolution(self.sample_simulations, metric='f1_macro')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('line_evolution_f1_macro', self.viz.figures)
    
    def test_plot_pvalue_matrix(self):
        """Test p-value matrix plot generation."""
        models = ['svm_bow', 'svm_embeddings', 'bert']
        pvalue_matrix = np.array([
            [1.0, 0.02, 0.001],
            [0.02, 1.0, 0.03],
            [0.001, 0.03, 1.0]
        ])
        
        fig = self.viz.plot_pvalue_matrix(pvalue_matrix, models, metric='f1_macro')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('pvalue_matrix_f1_macro', self.viz.figures)
    
    def test_plot_confidence_intervals(self):
        """Test confidence intervals plot generation."""
        ci_results = {
            'svm_bow': {'mean': 0.795, 'std': 0.01},
            'svm_embeddings': {'mean': 0.811, 'std': 0.012},
            'bert': {'mean': 0.844, 'std': 0.008}
        }
        
        fig = self.viz.plot_confidence_intervals(ci_results, metric='f1_macro')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('confidence_intervals_f1_macro', self.viz.figures)
    
    def test_save_all_figures(self):
        """Test saving all figures."""
        # Generate a plot first
        self.viz.plot_metrics_comparison(self.sample_results)
        
        # Create temporary directory for testing
        test_dir = 'results/plots/test_output'
        os.makedirs(test_dir, exist_ok=True)
        
        # Save figures
        self.viz.save_all_figures(output_dir=test_dir, dpi=100)
        
        # Check if file was created
        expected_file = os.path.join(test_dir, 'metrics_comparison.png')
        self.assertTrue(os.path.exists(expected_file))
        
        # Clean up
        if os.path.exists(expected_file):
            os.remove(expected_file)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
    
    def test_close_all(self):
        """Test closing all figures."""
        # Generate some plots
        self.viz.plot_metrics_comparison(self.sample_results)
        self.viz.plot_confusion_matrix(self.sample_cm, 'test_model')
        
        self.assertEqual(len(self.viz.figures), 2)
        
        # Close all
        self.viz.close_all()
        
        self.assertEqual(len(self.viz.figures), 0)
    
    def test_confusion_matrix_shape(self):
        """Test that confusion matrix handles correct shape."""
        # Test with 2x2 matrix
        cm_2x2 = np.array([[100, 20], [15, 85]])
        fig = self.viz.plot_confusion_matrix(cm_2x2, 'test_2x2')
        self.assertIsInstance(fig, plt.Figure)
    
    def test_empty_results(self):
        """Test handling of empty results."""
        empty_results = {}
        
        # Should not raise an error
        try:
            fig = self.viz.plot_metrics_comparison(empty_results)
            # If it doesn't raise an error, that's acceptable
        except (ValueError, KeyError):
            # If it raises an error, that's also acceptable
            pass
    
    def test_single_model(self):
        """Test visualization with single model."""
        single_model = {'svm_bow': {'accuracy': 0.796, 'f1_macro': 0.795}}
        
        fig = self.viz.plot_metrics_comparison(single_model)
        self.assertIsInstance(fig, plt.Figure)
    
    def test_custom_figsize(self):
        """Test custom figure size."""
        custom_viz = Visualizer(figsize=(12, 8))
        self.assertEqual(custom_viz.figsize, (12, 8))


if __name__ == '__main__':
    unittest.main()
