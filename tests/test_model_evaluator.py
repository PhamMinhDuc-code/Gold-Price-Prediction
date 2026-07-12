"""
Unit tests for ModelEvaluator class.

Tests cover:
- Metric calculations (MAE, RMSE, MAPE, R², directional accuracy)
- Edge cases and error handling
- Plot generation
- Report generation

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.model_evaluator import ModelEvaluator
from config import Config


class TestModelEvaluatorInit:
    """Test ModelEvaluator initialization."""
    
    def test_init_with_data(self):
        """Test initialization with data provided."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        assert evaluator.y_true is not None
        assert evaluator.y_pred is not None
        assert len(evaluator.y_true) == 5
        assert len(evaluator.y_pred) == 5
    
    def test_init_without_data(self):
        """Test initialization without data."""
        evaluator = ModelEvaluator()
        
        assert evaluator.y_true is None
        assert evaluator.y_pred is None
        assert evaluator.dates is None
        assert evaluator.metrics == {}
    
    def test_init_with_dates(self):
        """Test initialization with dates."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        dates = [datetime(2024, 1, i) for i in range(1, 4)]
        
        evaluator = ModelEvaluator(y_true, y_pred, dates)
        
        assert evaluator.dates is not None
        assert len(evaluator.dates) == 3
    
    def test_init_mismatched_shapes(self):
        """Test initialization with mismatched array shapes."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1])  # Different length
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            ModelEvaluator(y_true, y_pred)
    
    def test_set_data(self):
        """Test set_data method."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        
        evaluator.set_data(y_true, y_pred)
        
        assert evaluator.y_true is not None
        assert len(evaluator.y_true) == 3


class TestMetricCalculations:
    """Test metric calculation methods."""
    
    def test_calculate_mae_perfect_predictions(self):
        """Test MAE with perfect predictions (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        mae = evaluator.calculate_mae()
        
        assert mae == pytest.approx(0.0, abs=1e-10)
        assert evaluator.metrics['MAE'] == pytest.approx(0.0, abs=1e-10)
    
    def test_calculate_mae_known_values(self):
        """Test MAE with known values."""
        # MAE = (|1-2| + |2-3| + |3-4|) / 3 = 1.0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        mae = evaluator.calculate_mae()
        
        assert mae == pytest.approx(1.0, abs=1e-6)
    
    def test_calculate_rmse_perfect_predictions(self):
        """Test RMSE with perfect predictions (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        rmse = evaluator.calculate_rmse()
        
        assert rmse == pytest.approx(0.0, abs=1e-10)
    
    def test_calculate_rmse_known_values(self):
        """Test RMSE with known values."""
        # RMSE = sqrt((1² + 1² + 1²) / 3) = 1.0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        rmse = evaluator.calculate_rmse()
        
        assert rmse == pytest.approx(1.0, abs=1e-6)
    
    def test_calculate_mape_known_values(self):
        """Test MAPE with known values."""
        # MAPE = 100 * (|100-90|/100 + |200-180|/200) / 2
        #      = 100 * (0.1 + 0.1) / 2 = 10%
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 180.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        mape = evaluator.calculate_mape()
        
        assert mape == pytest.approx(10.0, abs=1e-6)
    
    def test_calculate_mape_zero_handling(self):
        """Test MAPE handles zero values with epsilon."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 90.0, 180.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        mape = evaluator.calculate_mape()
        
        # Should not raise division by zero error
        assert mape > 0  # Will be very large due to epsilon
    
    def test_calculate_r2_perfect_predictions(self):
        """Test R² with perfect predictions (should be 1.0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        r2 = evaluator.calculate_r2()
        
        assert r2 == pytest.approx(1.0, abs=1e-10)
    
    def test_calculate_r2_mean_predictions(self):
        """Test R² with predictions equal to mean (should be 0.0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Mean of y_true
        
        evaluator = ModelEvaluator(y_true, y_pred)
        r2 = evaluator.calculate_r2()
        
        assert r2 == pytest.approx(0.0, abs=1e-10)
    
    def test_calculate_r2_known_values(self):
        """Test R² with known values."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        
        # Manually calculated R² ≈ 0.9486
        evaluator = ModelEvaluator(y_true, y_pred)
        r2 = evaluator.calculate_r2()
        
        assert 0.9 < r2 < 1.0
    
    def test_calculate_directional_accuracy_perfect(self):
        """Test directional accuracy with perfect direction predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 3.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        accuracy = evaluator.calculate_directional_accuracy()
        
        assert accuracy == pytest.approx(100.0, abs=1e-6)
    
    def test_calculate_directional_accuracy_known_values(self):
        """Test directional accuracy with known values."""
        # True: 1→2 (up), 2→3 (up), 3→2 (down)
        # Pred: 1.1→2.1 (up), 2.1→2.9 (up), 2.9→2.2 (down)
        # All directions correct: 100%
        y_true = np.array([1.0, 2.0, 3.0, 2.0])
        y_pred = np.array([1.1, 2.1, 2.9, 2.2])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        accuracy = evaluator.calculate_directional_accuracy()
        
        assert accuracy == pytest.approx(100.0, abs=1e-6)
    
    def test_calculate_directional_accuracy_partial(self):
        """Test directional accuracy with partial correctness."""
        # True: 1→2 (up), 2→3 (up), 3→2 (down)
        # Pred: 1.1→2.1 (up), 2.1→1.9 (down), 1.9→1.5 (down)
        # Correct: 1st (up-up), incorrect: 2nd (up-down), correct: 3rd (down-down)
        # Accuracy: 2/3 = 66.67%
        y_true = np.array([1.0, 2.0, 3.0, 2.0])
        y_pred = np.array([1.1, 2.1, 1.9, 1.5])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        accuracy = evaluator.calculate_directional_accuracy()
        
        assert accuracy == pytest.approx(66.67, abs=0.1)
    
    def test_calculate_directional_accuracy_insufficient_samples(self):
        """Test directional accuracy with only 1 sample."""
        y_true = np.array([1.0])
        y_pred = np.array([1.1])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        accuracy = evaluator.calculate_directional_accuracy()
        
        assert accuracy == 0.0
    
    def test_calculate_all_metrics(self):
        """Test calculate_all_metrics method."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        metrics = evaluator.calculate_all_metrics()
        
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        assert 'Directional_Accuracy' in metrics
        
        # All metrics should be stored
        assert evaluator.metrics == metrics
    
    def test_metrics_without_data(self):
        """Test metrics calculation without data should raise error."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError):
            evaluator.calculate_mae()
        
        with pytest.raises(ValueError):
            evaluator.calculate_rmse()


class TestPlotGeneration:
    """Test plot generation methods."""
    
    def test_plot_residuals_basic(self):
        """Test basic residual plot generation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        fig = evaluator.plot_residuals(show=False)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert 'residuals' in evaluator.figures
        
        plt.close(fig)
    
    def test_plot_residuals_with_dates(self):
        """Test residual plot with dates."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        dates = [datetime(2024, 1, i) for i in range(1, 4)]
        
        evaluator = ModelEvaluator(y_true, y_pred, dates)
        fig = evaluator.plot_residuals(show=False)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_residuals_save(self, tmp_path):
        """Test saving residual plot to file."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        save_path = tmp_path / "residuals.png"
        fig = evaluator.plot_residuals(save_path=save_path, show=False)
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        
        plt.close(fig)
    
    def test_plot_predictions_vs_actual_basic(self):
        """Test basic predictions vs actual plot."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        fig = evaluator.plot_predictions_vs_actual(show=False)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert 'predictions_vs_actual' in evaluator.figures
        
        plt.close(fig)
    
    def test_plot_predictions_vs_actual_with_dates(self):
        """Test predictions vs actual plot with dates."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        dates = [datetime(2024, 1, i) for i in range(1, 4)]
        
        evaluator = ModelEvaluator(y_true, y_pred, dates)
        fig = evaluator.plot_predictions_vs_actual(show=False)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_predictions_vs_actual_save(self, tmp_path):
        """Test saving predictions plot to file."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        save_path = tmp_path / "predictions.png"
        fig = evaluator.plot_predictions_vs_actual(save_path=save_path, show=False)
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        
        plt.close(fig)
    
    def test_plot_without_data(self):
        """Test plotting without data should raise error."""
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError):
            evaluator.plot_residuals(show=False)
        
        with pytest.raises(ValueError):
            evaluator.plot_predictions_vs_actual(show=False)


class TestPerformanceReport:
    """Test performance report generation."""
    
    def test_generate_performance_report_basic(self, tmp_path):
        """Test basic performance report generation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        # Temporarily change reports directory
        original_reports_dir = Config.REPORTS_DIR
        Config.REPORTS_DIR = tmp_path
        
        try:
            report = evaluator.generate_performance_report(
                report_name="test_report",
                save_json=True,
                save_plots=True
            )
            
            # Check report structure
            assert 'report_name' in report
            assert 'generated_at' in report
            assert 'sample_count' in report
            assert 'metrics' in report
            assert 'summary' in report
            
            # Check metrics in summary
            assert 'mae' in report['summary']
            assert 'rmse' in report['summary']
            assert 'mape' in report['summary']
            assert 'r2' in report['summary']
            assert 'directional_accuracy' in report['summary']
            
            # Check sample count
            assert report['sample_count'] == 5
            
            # Check files were created
            json_path = tmp_path / "test_report.json"
            assert json_path.exists()
            
            residuals_path = tmp_path / "test_report_residuals.png"
            assert residuals_path.exists()
            
            predictions_path = tmp_path / "test_report_predictions.png"
            assert predictions_path.exists()
            
        finally:
            Config.REPORTS_DIR = original_reports_dir
    
    def test_generate_performance_report_with_dates(self, tmp_path):
        """Test performance report with dates."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        dates = [datetime(2024, 1, i) for i in range(1, 4)]
        
        evaluator = ModelEvaluator(y_true, y_pred, dates)
        
        # Temporarily change reports directory
        original_reports_dir = Config.REPORTS_DIR
        Config.REPORTS_DIR = tmp_path
        
        try:
            report = evaluator.generate_performance_report(
                report_name="test_report_dates",
                save_json=True,
                save_plots=False
            )
            
            assert 'date_range' in report
            assert report['date_range'] is not None
            assert 'start' in report['date_range']
            assert 'end' in report['date_range']
            
        finally:
            Config.REPORTS_DIR = original_reports_dir
    
    def test_generate_performance_report_json_content(self, tmp_path):
        """Test JSON report content validity."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        # Temporarily change reports directory
        original_reports_dir = Config.REPORTS_DIR
        Config.REPORTS_DIR = tmp_path
        
        try:
            report = evaluator.generate_performance_report(
                report_name="test_json",
                save_json=True,
                save_plots=False
            )
            
            # Load and verify JSON
            json_path = tmp_path / "test_json.json"
            with open(json_path, 'r') as f:
                loaded_report = json.load(f)
            
            assert loaded_report['report_name'] == "test_json"
            assert loaded_report['sample_count'] == 5
            assert 'metrics' in loaded_report
            
        finally:
            Config.REPORTS_DIR = original_reports_dir
    
    def test_generate_performance_report_no_save(self):
        """Test report generation without saving files."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        report = evaluator.generate_performance_report(
            report_name="no_save_test",
            save_json=False,
            save_plots=False
        )
        
        # Report should still be generated
        assert report is not None
        assert 'metrics' in report
        
        # But no file paths should be in report
        assert 'json_path' not in report
        assert 'residual_plot_path' not in report
        assert 'predictions_plot_path' not in report


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test with single sample."""
        y_true = np.array([1.0])
        y_pred = np.array([1.1])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        # MAE and RMSE should work
        mae = evaluator.calculate_mae()
        assert mae == pytest.approx(0.1, abs=1e-6)
        
        rmse = evaluator.calculate_rmse()
        assert rmse == pytest.approx(0.1, abs=1e-6)
        
        # Directional accuracy needs at least 2 samples
        accuracy = evaluator.calculate_directional_accuracy()
        assert accuracy == 0.0
    
    def test_large_errors(self):
        """Test with large prediction errors."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([150.0, 250.0, 350.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        mae = evaluator.calculate_mae()
        assert mae == pytest.approx(50.0, abs=1e-6)
        
        rmse = evaluator.calculate_rmse()
        assert rmse == pytest.approx(50.0, abs=1e-6)
    
    def test_negative_values(self):
        """Test with negative values."""
        y_true = np.array([-1.0, -2.0, -3.0])
        y_pred = np.array([-1.1, -2.1, -2.9])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        # Should work without errors
        mae = evaluator.calculate_mae()
        assert mae > 0
        
        r2 = evaluator.calculate_r2()
        assert r2 >= 0
    
    def test_constant_predictions(self):
        """Test with constant predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        
        evaluator = ModelEvaluator(y_true, y_pred)
        
        # R² should be 0 (as bad as predicting mean)
        r2 = evaluator.calculate_r2()
        assert r2 == pytest.approx(0.0, abs=1e-10)
        
        # Directional accuracy should be 0
        accuracy = evaluator.calculate_directional_accuracy()
        assert accuracy == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
