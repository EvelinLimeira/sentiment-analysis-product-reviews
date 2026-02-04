"""
Unit tests for the simulation runner module.

Tests the SimulationRunner class and related functionality for running
multiple simulations with different random seeds.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.config import ExperimentConfig
from src.simulation_runner import SimulationRunner, SimulationResult, run_all_simulations


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def quick_config():
    """Create a quick test configuration."""
    return ExperimentConfig(
        dataset_name='amazon_reviews',
        num_simulations=2,
        tfidf_max_features=500,
        bert_epochs=1,
        bert_batch_size=8
    )


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_simulation_result_creation(self):
        """Test creating a SimulationResult."""
        result = SimulationResult(
            simulation_id=0,
            model_name='svm_bow',
            random_seed=42,
            accuracy=0.85,
            precision_macro=0.84,
            recall_macro=0.83,
            f1_macro=0.835,
            f1_weighted=0.836,
            training_time=10.5,
            inference_time=0.5
        )
        
        assert result.simulation_id == 0
        assert result.model_name == 'svm_bow'
        assert result.random_seed == 42
        assert result.accuracy == 0.85
        assert result.training_time == 10.5
    
    def test_simulation_result_to_dict(self):
        """Test converting SimulationResult to dictionary."""
        from dataclasses import asdict
        
        result = SimulationResult(
            simulation_id=0,
            model_name='svm_bow',
            random_seed=42,
            accuracy=0.85,
            precision_macro=0.84,
            recall_macro=0.83,
            f1_macro=0.835,
            f1_weighted=0.836,
            training_time=10.5,
            inference_time=0.5
        )
        
        result_dict = asdict(result)
        
        assert isinstance(result_dict, dict)
        assert result_dict['simulation_id'] == 0
        assert result_dict['model_name'] == 'svm_bow'
        assert result_dict['accuracy'] == 0.85


class TestSimulationRunner:
    """Tests for SimulationRunner class."""
    
    def test_initialization(self, quick_config, temp_output_dir):
        """Test SimulationRunner initialization."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        assert runner.config == quick_config
        assert runner.output_dir == Path(temp_output_dir)
        assert runner.output_dir.exists()
    
    def test_run_single_simulation_svm_bow(self, quick_config, temp_output_dir):
        """Test running a single SVM+BoW simulation."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        result = runner.run_single_simulation(
            model_name='svm_bow',
            simulation_id=0,
            random_seed=42
        )
        
        assert isinstance(result, SimulationResult)
        assert result.simulation_id == 0
        assert result.model_name == 'svm_bow'
        assert result.random_seed == 42
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.f1_macro <= 1.0
        assert result.training_time > 0
        assert result.inference_time > 0
    
    def test_run_single_simulation_invalid_model(self, quick_config, temp_output_dir):
        """Test running simulation with invalid model name."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        with pytest.raises(ValueError, match="Unknown model name"):
            runner.run_single_simulation(
                model_name='invalid_model',
                simulation_id=0,
                random_seed=42
            )
    
    def test_run_simulations_creates_csv(self, quick_config, temp_output_dir):
        """Test that run_simulations creates CSV files."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        results = runner.run_simulations(
            model_names=['svm_bow'],
            base_seed=42
        )
        
        # Check results dictionary
        assert 'svm_bow' in results
        assert isinstance(results['svm_bow'], pd.DataFrame)
        assert len(results['svm_bow']) == quick_config.num_simulations
        
        # Check CSV file was created
        csv_file = Path(temp_output_dir) / 'svm_bow_simulations.csv'
        assert csv_file.exists()
        
        # Verify CSV content
        df = pd.read_csv(csv_file)
        assert len(df) == quick_config.num_simulations
        assert 'simulation_id' in df.columns
        assert 'model_name' in df.columns
        assert 'random_seed' in df.columns
        assert 'accuracy' in df.columns
        assert 'f1_macro' in df.columns
    
    def test_run_simulations_different_seeds(self, quick_config, temp_output_dir):
        """Test that simulations use different random seeds."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        results = runner.run_simulations(
            model_names=['svm_bow'],
            base_seed=100
        )
        
        df = results['svm_bow']
        
        # Check that seeds are different
        seeds = df['random_seed'].tolist()
        assert len(seeds) == len(set(seeds))  # All unique
        assert seeds[0] == 100  # First seed is base_seed
        assert seeds[1] == 101  # Second seed is base_seed + 1
    
    def test_calculate_confidence_intervals(self, quick_config, temp_output_dir):
        """Test confidence interval calculation."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        # Create sample data
        df = pd.DataFrame({
            'f1_macro': [0.80, 0.82, 0.81, 0.83, 0.79]
        })
        
        mean, lower, upper = runner.calculate_confidence_intervals(
            df, metric='f1_macro', confidence=0.95
        )
        
        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < mean < upper
        assert 0.0 <= mean <= 1.0
    
    def test_get_summary_table(self, quick_config, temp_output_dir):
        """Test summary table generation."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        # Create sample results
        results = {
            'svm_bow': pd.DataFrame({
                'accuracy': [0.80, 0.82],
                'precision_macro': [0.79, 0.81],
                'recall_macro': [0.78, 0.80],
                'f1_macro': [0.785, 0.805],
                'f1_weighted': [0.786, 0.806],
                'training_time': [10.0, 11.0],
                'inference_time': [0.5, 0.6]
            }),
            'svm_embeddings': pd.DataFrame({
                'accuracy': [0.75, 0.77],
                'precision_macro': [0.74, 0.76],
                'recall_macro': [0.73, 0.75],
                'f1_macro': [0.735, 0.755],
                'f1_weighted': [0.736, 0.756],
                'training_time': [15.0, 16.0],
                'inference_time': [0.7, 0.8]
            })
        }
        
        summary = runner.get_summary_table(results)
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'model' in summary.columns
        assert 'accuracy' in summary.columns
        assert 'f1_macro' in summary.columns
        
        # Check format (mean ± std)
        assert '±' in summary.iloc[0]['accuracy']
    
    def test_log_summary_statistics(self, quick_config, temp_output_dir, caplog):
        """Test summary statistics logging."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        df = pd.DataFrame({
            'accuracy': [0.80, 0.82, 0.81],
            'f1_macro': [0.79, 0.81, 0.80],
            'training_time': [10.0, 11.0, 10.5]
        })
        
        runner._log_summary_statistics('test_model', df)
        
        # Check that logging occurred
        assert 'Summary statistics for test_model' in caplog.text
        assert 'accuracy' in caplog.text
        assert 'f1_macro' in caplog.text


class TestRunAllSimulations:
    """Tests for run_all_simulations convenience function."""
    
    def test_run_all_simulations_single_model(self, quick_config, temp_output_dir):
        """Test running simulations for a single model."""
        results = run_all_simulations(
            config=quick_config,
            models=['svm_bow'],
            base_seed=42,
            output_dir=temp_output_dir
        )
        
        assert 'svm_bow' in results
        assert isinstance(results['svm_bow'], pd.DataFrame)
        
        # Check summary file was created
        summary_file = Path(temp_output_dir) / 'summary_statistics.csv'
        assert summary_file.exists()
    
    def test_run_all_simulations_creates_summary(self, quick_config, temp_output_dir):
        """Test that summary statistics file is created."""
        results = run_all_simulations(
            config=quick_config,
            models=['svm_bow'],
            base_seed=42,
            output_dir=temp_output_dir
        )
        
        summary_file = Path(temp_output_dir) / 'summary_statistics.csv'
        assert summary_file.exists()
        
        # Verify summary content
        summary_df = pd.read_csv(summary_file)
        assert 'model' in summary_df.columns
        assert len(summary_df) == 1
        assert summary_df.iloc[0]['model'] == 'svm_bow'


class TestSimulationResultFormat:
    """Tests for simulation result format compatibility."""
    
    def test_csv_format_matches_statistical_validator(self, quick_config, temp_output_dir):
        """Test that CSV format matches what StatisticalValidator expects."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        results = runner.run_simulations(
            model_names=['svm_bow'],
            base_seed=42
        )
        
        csv_file = Path(temp_output_dir) / 'svm_bow_simulations.csv'
        df = pd.read_csv(csv_file)
        
        # Check required columns for StatisticalValidator
        required_columns = [
            'simulation_id',
            'model_name',
            'random_seed',
            'accuracy',
            'precision_macro',
            'recall_macro',
            'f1_macro',
            'f1_weighted',
            'training_time',
            'inference_time'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data types
        assert df['simulation_id'].dtype in [np.int64, np.int32]
        assert df['model_name'].dtype == object
        assert df['random_seed'].dtype in [np.int64, np.int32]
        assert df['accuracy'].dtype in [np.float64, np.float32]
        assert df['f1_macro'].dtype in [np.float64, np.float32]
    
    def test_metrics_in_valid_range(self, quick_config, temp_output_dir):
        """Test that all metrics are in valid ranges."""
        runner = SimulationRunner(quick_config, output_dir=temp_output_dir)
        
        results = runner.run_simulations(
            model_names=['svm_bow'],
            base_seed=42
        )
        
        df = results['svm_bow']
        
        # Check metric ranges
        metric_columns = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
        
        for col in metric_columns:
            assert (df[col] >= 0.0).all(), f"{col} has values < 0"
            assert (df[col] <= 1.0).all(), f"{col} has values > 1"
        
        # Check timing values are positive
        assert (df['training_time'] > 0).all()
        assert (df['inference_time'] > 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
