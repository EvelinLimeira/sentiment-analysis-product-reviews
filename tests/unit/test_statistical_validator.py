"""
Unit tests for StatisticalValidator module.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

# Set matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')

from src.statistical_validator import StatisticalValidator


@pytest.fixture
def temp_results_dir():
    """Create temporary results directory structure."""
    temp_dir = tempfile.mkdtemp()
    simulations_dir = Path(temp_dir) / 'results' / 'simulations'
    plots_dir = Path(temp_dir) / 'results' / 'plots' / 'statistical'
    tests_dir = Path(temp_dir) / 'results' / 'statistical_tests'
    
    simulations_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # Cleanup
    os.chdir(original_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_simulation_data():
    """Create sample simulation data for testing."""
    np.random.seed(42)
    
    # Model 1: Better performance
    model1_data = pd.DataFrame({
        'simulation_id': range(10),
        'accuracy': np.random.normal(0.85, 0.02, 10),
        'precision': np.random.normal(0.84, 0.02, 10),
        'recall': np.random.normal(0.86, 0.02, 10),
        'f1_score': np.random.normal(0.85, 0.02, 10),
        'training_time': np.random.normal(100, 10, 10),
        'inference_time': np.random.normal(5, 0.5, 10)
    })
    
    # Model 2: Slightly worse performance
    model2_data = pd.DataFrame({
        'simulation_id': range(10),
        'accuracy': np.random.normal(0.80, 0.02, 10),
        'precision': np.random.normal(0.79, 0.02, 10),
        'recall': np.random.normal(0.81, 0.02, 10),
        'f1_score': np.random.normal(0.80, 0.02, 10),
        'training_time': np.random.normal(120, 10, 10),
        'inference_time': np.random.normal(6, 0.5, 10)
    })
    
    # Model 3: Similar to model 2
    model3_data = pd.DataFrame({
        'simulation_id': range(10),
        'accuracy': np.random.normal(0.81, 0.02, 10),
        'precision': np.random.normal(0.80, 0.02, 10),
        'recall': np.random.normal(0.82, 0.02, 10),
        'f1_score': np.random.normal(0.81, 0.02, 10),
        'training_time': np.random.normal(115, 10, 10),
        'inference_time': np.random.normal(5.5, 0.5, 10)
    })
    
    return {
        'model1': model1_data,
        'model2': model2_data,
        'model3': model3_data
    }


@pytest.fixture
def validator_with_data(temp_results_dir, sample_simulation_data):
    """Create validator with sample data files."""
    # Save sample data to CSV files
    for model_name, data in sample_simulation_data.items():
        data.to_csv(f'results/simulations/{model_name}_simulations.csv', index=False)
    
    validator = StatisticalValidator(alpha=0.05)
    return validator


class TestStatisticalValidatorInit:
    """Test StatisticalValidator initialization."""
    
    def test_init_default_alpha(self):
        """Test initialization with default alpha."""
        validator = StatisticalValidator()
        assert validator.alpha == 0.05
    
    def test_init_custom_alpha(self):
        """Test initialization with custom alpha."""
        validator = StatisticalValidator(alpha=0.01)
        assert validator.alpha == 0.01
    
    def test_alpha_range(self):
        """Test that alpha is in valid range."""
        validator = StatisticalValidator(alpha=0.05)
        assert 0 < validator.alpha < 1


class TestLoadSimulations:
    """Test load_simulations method."""
    
    def test_load_simulations_success(self, validator_with_data):
        """Test successful loading of simulation data."""
        df = validator_with_data.load_simulations('model1')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'f1_score' in df.columns
        assert 'accuracy' in df.columns
    
    def test_load_simulations_file_not_found(self, temp_results_dir):
        """Test loading non-existent simulation file."""
        validator = StatisticalValidator()
        with pytest.raises(FileNotFoundError):
            validator.load_simulations('nonexistent_model')


class TestShapiroNormality:
    """Test shapiro_normality method."""
    
    def test_shapiro_normality_structure(self, validator_with_data):
        """Test that shapiro test returns correct structure."""
        result = validator_with_data.shapiro_normality('model1', 'f1_score')
        
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'metric' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'is_normal' in result
    
    def test_shapiro_normality_values(self, validator_with_data):
        """Test that shapiro test returns valid values."""
        result = validator_with_data.shapiro_normality('model1', 'f1_score')
        
        assert result['model'] == 'model1'
        assert result['metric'] == 'f1_score'
        assert isinstance(result['statistic'], (float, np.floating))
        assert isinstance(result['p_value'], (float, np.floating))
        assert isinstance(result['is_normal'], (bool, np.bool_))
        
        # P-value should be in [0, 1]
        assert 0 <= result['p_value'] <= 1
    
    def test_shapiro_normality_different_metrics(self, validator_with_data):
        """Test shapiro test with different metrics."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            result = validator_with_data.shapiro_normality('model1', metric)
            assert result['metric'] == metric
            assert 0 <= result['p_value'] <= 1


class TestKruskalWallisMultiple:
    """Test kruskal_wallis_multiple method."""
    
    def test_kruskal_wallis_structure(self, validator_with_data):
        """Test that Kruskal-Wallis test returns correct structure."""
        model_names = ['model1', 'model2', 'model3']
        result = validator_with_data.kruskal_wallis_multiple(model_names, 'f1_score')
        
        assert isinstance(result, dict)
        assert 'models' in result
        assert 'metric' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
    
    def test_kruskal_wallis_values(self, validator_with_data):
        """Test that Kruskal-Wallis test returns valid values."""
        model_names = ['model1', 'model2', 'model3']
        result = validator_with_data.kruskal_wallis_multiple(model_names, 'f1_score')
        
        assert result['models'] == model_names
        assert result['metric'] == 'f1_score'
        assert isinstance(result['statistic'], (float, np.floating))
        assert isinstance(result['p_value'], (float, np.floating))
        assert isinstance(result['significant'], (bool, np.bool_))
        
        # P-value should be in [0, 1]
        assert 0 <= result['p_value'] <= 1
    
    def test_kruskal_wallis_two_models(self, validator_with_data):
        """Test Kruskal-Wallis with only two models."""
        model_names = ['model1', 'model2']
        result = validator_with_data.kruskal_wallis_multiple(model_names, 'accuracy')
        
        assert len(result['models']) == 2
        assert 0 <= result['p_value'] <= 1
    
    def test_kruskal_wallis_significance(self, validator_with_data):
        """Test that significance is correctly determined."""
        model_names = ['model1', 'model2', 'model3']
        result = validator_with_data.kruskal_wallis_multiple(model_names, 'f1_score')
        
        # Significance should match p_value < alpha
        expected_significant = result['p_value'] < validator_with_data.alpha
        assert result['significant'] == expected_significant


class TestWilcoxonPairwise:
    """Test wilcoxon_pairwise method."""
    
    def test_wilcoxon_pairwise_structure(self, validator_with_data):
        """Test that Wilcoxon test returns correct structure."""
        result = validator_with_data.wilcoxon_pairwise('model1', 'model2', 'f1_score')
        
        assert isinstance(result, dict)
        assert 'model1' in result
        assert 'model2' in result
        assert 'metric' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'winner' in result
        assert 'model1_median' in result
        assert 'model2_median' in result
    
    def test_wilcoxon_pairwise_values(self, validator_with_data):
        """Test that Wilcoxon test returns valid values."""
        result = validator_with_data.wilcoxon_pairwise('model1', 'model2', 'f1_score')
        
        assert result['model1'] == 'model1'
        assert result['model2'] == 'model2'
        assert result['metric'] == 'f1_score'
        assert isinstance(result['statistic'], (float, np.floating))
        assert isinstance(result['p_value'], (float, np.floating))
        assert isinstance(result['significant'], bool)
        
        # P-value should be in [0, 1]
        assert 0 <= result['p_value'] <= 1
        
        # Medians should be valid floats
        assert isinstance(result['model1_median'], (float, np.floating))
        assert isinstance(result['model2_median'], (float, np.floating))
    
    def test_wilcoxon_pairwise_winner_logic(self, validator_with_data):
        """Test that winner is correctly determined."""
        result = validator_with_data.wilcoxon_pairwise('model1', 'model2', 'f1_score')
        
        if result['significant']:
            # Winner should be the model with higher median
            if result['model1_median'] > result['model2_median']:
                assert result['winner'] == 'model1'
            else:
                assert result['winner'] == 'model2'
        else:
            assert result['winner'] == "No significant difference"
    
    def test_wilcoxon_pairwise_symmetric(self, validator_with_data):
        """Test that Wilcoxon test is symmetric (same p-value regardless of order)."""
        result1 = validator_with_data.wilcoxon_pairwise('model1', 'model2', 'f1_score')
        result2 = validator_with_data.wilcoxon_pairwise('model2', 'model1', 'f1_score')
        
        # P-values should be the same
        assert abs(result1['p_value'] - result2['p_value']) < 1e-10


class TestGeneratePvalueMatrix:
    """Test generate_pvalue_matrix method."""
    
    def test_pvalue_matrix_shape(self, validator_with_data):
        """Test that p-value matrix has correct shape."""
        model_names = ['model1', 'model2', 'model3']
        matrix = validator_with_data.generate_pvalue_matrix(model_names, 'f1_score')
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
    
    def test_pvalue_matrix_diagonal(self, validator_with_data):
        """Test that diagonal elements are 1.0."""
        model_names = ['model1', 'model2', 'model3']
        matrix = validator_with_data.generate_pvalue_matrix(model_names, 'f1_score')
        
        for i in range(len(model_names)):
            assert matrix[i, i] == 1.0
    
    def test_pvalue_matrix_range(self, validator_with_data):
        """Test that all p-values are in [0, 1]."""
        model_names = ['model1', 'model2', 'model3']
        matrix = validator_with_data.generate_pvalue_matrix(model_names, 'f1_score')
        
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 1)
    
    def test_pvalue_matrix_symmetry(self, validator_with_data):
        """Test that p-value matrix is symmetric."""
        model_names = ['model1', 'model2', 'model3']
        matrix = validator_with_data.generate_pvalue_matrix(model_names, 'f1_score')
        
        # Matrix should be symmetric (p-value for A vs B = p-value for B vs A)
        assert np.allclose(matrix, matrix.T)
    
    def test_pvalue_matrix_plot_created(self, validator_with_data):
        """Test that plot file is created."""
        model_names = ['model1', 'model2']
        validator_with_data.generate_pvalue_matrix(model_names, 'accuracy')
        
        plot_path = 'results/plots/statistical/pvalue_matrix_accuracy.png'
        assert os.path.exists(plot_path)


class TestGenerateReport:
    """Test generate_report method."""
    
    def test_generate_report_returns_string(self, validator_with_data):
        """Test that generate_report returns a string."""
        model_names = ['model1', 'model2']
        metrics = ['accuracy', 'f1_score']
        
        report = validator_with_data.generate_report(model_names, metrics)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_generate_report_contains_headers(self, validator_with_data):
        """Test that report contains expected headers."""
        model_names = ['model1', 'model2']
        metrics = ['accuracy', 'f1_score']
        
        report = validator_with_data.generate_report(model_names, metrics)
        
        assert "STATISTICAL VALIDATION REPORT" in report
        assert "KRUSKAL-WALLIS TEST" in report
        assert "WILCOXON PAIRED TESTS" in report
    
    def test_generate_report_contains_model_names(self, validator_with_data):
        """Test that report contains model names."""
        model_names = ['model1', 'model2', 'model3']
        metrics = ['f1_score']
        
        report = validator_with_data.generate_report(model_names, metrics)
        
        for model_name in model_names:
            assert model_name in report
    
    def test_generate_report_contains_metrics(self, validator_with_data):
        """Test that report contains metrics."""
        model_names = ['model1', 'model2']
        metrics = ['accuracy', 'f1_score']
        
        report = validator_with_data.generate_report(model_names, metrics)
        
        for metric in metrics:
            assert metric.upper() in report
    
    def test_generate_report_file_created(self, validator_with_data):
        """Test that report file is created."""
        model_names = ['model1', 'model2']
        metrics = ['accuracy']
        
        validator_with_data.generate_report(model_names, metrics)
        
        report_path = 'results/statistical_tests/statistical_report.txt'
        assert os.path.exists(report_path)
    
    def test_generate_report_file_content(self, validator_with_data):
        """Test that report file contains correct content."""
        model_names = ['model1', 'model2']
        metrics = ['f1_score']
        
        report = validator_with_data.generate_report(model_names, metrics)
        
        # Read the file
        with open('results/statistical_tests/statistical_report.txt', 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # File content should match returned report
        assert file_content == report
    
    def test_generate_report_alpha_in_report(self, validator_with_data):
        """Test that alpha value is mentioned in report."""
        model_names = ['model1', 'model2']
        metrics = ['accuracy']
        
        report = validator_with_data.generate_report(model_names, metrics)
        
        assert str(validator_with_data.alpha) in report


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_model_kruskal_wallis(self, validator_with_data):
        """Test Kruskal-Wallis with single model (should handle gracefully)."""
        # This should raise an error or handle gracefully
        # scipy.stats.kruskal requires at least 2 groups
        with pytest.raises((ValueError, TypeError, IndexError)):
            validator_with_data.kruskal_wallis_multiple(['model1'], 'f1_score')
    
    def test_identical_data_wilcoxon(self, temp_results_dir):
        """Test Wilcoxon with identical data."""
        # Create identical data
        data = pd.DataFrame({
            'simulation_id': range(10),
            'f1_score': [0.85] * 10
        })
        
        data.to_csv('results/simulations/model_a_simulations.csv', index=False)
        data.to_csv('results/simulations/model_b_simulations.csv', index=False)
        
        validator = StatisticalValidator()
        
        # Wilcoxon with identical data should handle gracefully
        # (may raise warning or return specific p-value)
        result = validator.wilcoxon_pairwise('model_a', 'model_b', 'f1_score')
        
        # Should still return valid structure
        assert 'p_value' in result
        assert result['winner'] == "No significant difference"


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self, validator_with_data):
        """Test complete statistical validation workflow."""
        model_names = ['model1', 'model2', 'model3']
        metrics = ['accuracy', 'f1_score']
        
        # 1. Test normality for each model
        for model in model_names:
            for metric in metrics:
                result = validator_with_data.shapiro_normality(model, metric)
                assert 0 <= result['p_value'] <= 1
        
        # 2. Test Kruskal-Wallis for multiple groups
        for metric in metrics:
            result = validator_with_data.kruskal_wallis_multiple(model_names, metric)
            assert 0 <= result['p_value'] <= 1
        
        # 3. Test pairwise comparisons
        for metric in metrics:
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    result = validator_with_data.wilcoxon_pairwise(model1, model2, metric)
                    assert 0 <= result['p_value'] <= 1
        
        # 4. Generate p-value matrix
        for metric in metrics:
            matrix = validator_with_data.generate_pvalue_matrix(model_names, metric)
            assert matrix.shape == (3, 3)
        
        # 5. Generate complete report
        report = validator_with_data.generate_report(model_names, metrics)
        assert len(report) > 0
        assert os.path.exists('results/statistical_tests/statistical_report.txt')
