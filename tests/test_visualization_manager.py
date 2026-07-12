"""
Unit Tests for VisualizationManager Module

Tests visualization functionality including:
- Time series plotting with predictions and confidence intervals
- Feature importance plotting
- Economic indicators overlay plotting
- Training history plotting
- Comprehensive prediction report generation

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import json
import tempfile
import shutil

from src.visualization_manager import VisualizationManager
from config import Config


class TestVisualizationManager(unittest.TestCase):
    """Test cases for VisualizationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Initialize visualization manager
        self.viz_manager = VisualizationManager()
        
        # Create sample data
        np.random.seed(42)
        self.n_samples = 100
        
        # Sample dates
        self.dates = pd.date_range(start='2023-01-01', periods=self.n_samples, freq='D')
        
        # Sample actual prices
        self.actual_prices = 1800 + np.cumsum(np.random.randn(self.n_samples) * 5)
        
        # Sample predictions (with some noise)
        self.predictions = self.actual_prices + np.random.randn(self.n_samples) * 10
        
        # Sample confidence intervals
        self.confidence_intervals = np.column_stack([
            self.predictions - 20,  # Lower bound
            self.predictions + 20   # Upper bound
        ])
        
        # Sample feature importance data
        self.feature_names = [f'Feature_{i}' for i in range(20)]
        self.feature_importances = np.random.rand(20)
        
        # Sample training history
        self.training_history = {
            'loss': [0.5, 0.4, 0.35, 0.32, 0.3, 0.29, 0.28, 0.27, 0.27, 0.27],
            'val_loss': [0.55, 0.45, 0.38, 0.35, 0.33, 0.32, 0.33, 0.34, 0.35, 0.36]
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        # Close all matplotlib figures
        plt.close('all')
    
    def test_initialization(self):
        """Test VisualizationManager initialization."""
        # Requirements: 9.1
        
        viz = VisualizationManager()
        
        # Verify manager is initialized
        self.assertIsNotNone(viz)
        self.assertIsInstance(viz.figures, dict)
        self.assertEqual(len(viz.figures), 0)
    
    def test_plot_time_series_with_predictions_basic(self):
        """Test basic time series plotting with predictions."""
        # Requirements: 9.1
        
        fig = self.viz_manager.plot_time_series_with_predictions(
            actual=self.actual_prices,
            predictions=self.predictions,
            dates=self.dates,
            show=False
        )
        
        # Verify figure is created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify figure is stored
        self.assertIn('time_series_predictions', self.viz_manager.figures)
    
    def test_plot_time_series_with_confidence_intervals(self):
        """Test time series plotting with confidence intervals."""
        # Requirements: 9.2
        
        fig = self.viz_manager.plot_time_series_with_predictions(
            actual=self.actual_prices,
            predictions=self.predictions,
            confidence_intervals=self.confidence_intervals,
            dates=self.dates,
            show=False
        )
        
        # Verify figure is created
        self.assertIsNotNone(fig)
        
        # Verify axes exist
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
        
        # Verify at least 3 lines/collections (actual, predicted, confidence band)
        ax = axes[0]
        self.assertGreaterEqual(len(ax.lines), 2)  # At least actual and predicted
    
    def test_plot_time_series_save(self):
        """Test saving time series plot to file."""
        # Requirements: 9.1
        
        save_path = self.test_dir / "test_time_series.png"
        
        fig = self.viz_manager.plot_time_series_with_predictions(
            actual=self.actual_prices,
            predictions=self.predictions,
            save_path=save_path,
            show=False
        )
        
        # Verify file is created
        self.assertTrue(save_path.exists())
        self.assertGreater(save_path.stat().st_size, 0)
    
    def test_plot_time_series_dataframe_input(self):
        """Test time series plotting with DataFrame input."""
        # Requirements: 9.1
        
        actual_df = pd.DataFrame({'Close': self.actual_prices})
        pred_df = pd.DataFrame({'Prediction': self.predictions})
        
        fig = self.viz_manager.plot_time_series_with_predictions(
            actual=actual_df,
            predictions=pred_df,
            show=False
        )
        
        # Verify figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        # Requirements: 9.3
        
        # Create mock model with feature_importances_ attribute
        class MockModel:
            def __init__(self, importances):
                self.feature_importances_ = importances
        
        mock_model = MockModel(self.feature_importances)
        
        fig = self.viz_manager.plot_feature_importance(
            model=mock_model,
            feature_names=self.feature_names,
            top_n=10,
            show=False
        )
        
        # Verify figure is created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify figure is stored
        self.assertIn('feature_importance', self.viz_manager.figures)
        
        # Verify axes exist
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
    
    def test_plot_feature_importance_top_n(self):
        """Test feature importance plotting with top N selection."""
        # Requirements: 9.3
        
        class MockModel:
            def __init__(self, importances):
                self.feature_importances_ = importances
        
        mock_model = MockModel(self.feature_importances)
        
        top_n = 5
        fig = self.viz_manager.plot_feature_importance(
            model=mock_model,
            feature_names=self.feature_names,
            top_n=top_n,
            show=False
        )
        
        # Verify correct number of features displayed
        ax = fig.get_axes()[0]
        self.assertEqual(len(ax.get_yticklabels()), top_n)
    
    def test_plot_feature_importance_no_attribute(self):
        """Test feature importance plotting with model lacking feature_importances_."""
        # Requirements: 9.3
        
        class MockModel:
            pass
        
        mock_model = MockModel()
        
        # Should raise AttributeError
        with self.assertRaises(AttributeError):
            self.viz_manager.plot_feature_importance(
                model=mock_model,
                feature_names=self.feature_names,
                show=False
            )
    
    def test_plot_indicators_overlay(self):
        """Test economic indicators overlay plotting."""
        # Requirements: 9.4
        
        # Create sample gold prices DataFrame
        gold_df = pd.DataFrame({
            'Date': self.dates,
            'Close': self.actual_prices
        })
        
        # Create sample indicators
        indicators = {
            'DXY': pd.DataFrame({
                'Date': self.dates,
                'Value': 100 + np.random.randn(self.n_samples) * 2
            }),
            'Oil': pd.DataFrame({
                'Date': self.dates,
                'Value': 80 + np.random.randn(self.n_samples) * 5
            })
        }
        
        fig = self.viz_manager.plot_indicators_overlay(
            gold_prices=gold_df,
            indicators=indicators,
            show=False
        )
        
        # Verify figure is created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify correct number of subplots (1 for gold + 1 per indicator)
        axes = fig.get_axes()
        expected_subplots = 1 + len(indicators)
        self.assertEqual(len(axes), expected_subplots)
        
        # Verify figure is stored
        self.assertIn('indicators_overlay', self.viz_manager.figures)
    
    def test_plot_indicators_overlay_save(self):
        """Test saving indicators overlay plot."""
        # Requirements: 9.4
        
        gold_df = pd.DataFrame({
            'Date': self.dates,
            'Close': self.actual_prices
        })
        
        indicators = {
            'DXY': pd.DataFrame({
                'Date': self.dates,
                'Value': 100 + np.random.randn(self.n_samples)
            })
        }
        
        save_path = self.test_dir / "test_indicators.png"
        
        fig = self.viz_manager.plot_indicators_overlay(
            gold_prices=gold_df,
            indicators=indicators,
            save_path=save_path,
            show=False
        )
        
        # Verify file is created
        self.assertTrue(save_path.exists())
        self.assertGreater(save_path.stat().st_size, 0)
    
    def test_plot_training_history(self):
        """Test training history plotting."""
        # Requirements: 5.6
        
        fig = self.viz_manager.plot_training_history(
            history=self.training_history,
            show=False
        )
        
        # Verify figure is created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Verify figure is stored
        self.assertIn('training_history', self.viz_manager.figures)
        
        # Verify axes exist
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
        
        # Verify at least 2 lines (training and validation loss)
        ax = axes[0]
        self.assertGreaterEqual(len(ax.lines), 2)
    
    def test_plot_training_history_no_validation(self):
        """Test training history plotting without validation loss."""
        # Requirements: 5.6
        
        history_no_val = {'loss': self.training_history['loss']}
        
        fig = self.viz_manager.plot_training_history(
            history=history_no_val,
            show=False
        )
        
        # Verify figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_training_history_missing_loss(self):
        """Test training history plotting with missing loss key."""
        # Requirements: 5.6
        
        invalid_history = {'val_loss': [0.5, 0.4, 0.3]}
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.viz_manager.plot_training_history(
                history=invalid_history,
                show=False
            )
    
    def test_plot_training_history_save(self):
        """Test saving training history plot."""
        # Requirements: 5.6
        
        save_path = self.test_dir / "test_training_history.png"
        
        fig = self.viz_manager.plot_training_history(
            history=self.training_history,
            save_path=save_path,
            show=False
        )
        
        # Verify file is created
        self.assertTrue(save_path.exists())
        self.assertGreater(save_path.stat().st_size, 0)

    
    def test_create_prediction_report_basic(self):
        """Test basic prediction report creation."""
        # Requirements: 9.5
        
        # Create sample predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': self.dates,
            'Actual': self.actual_prices,
            'Predicted': self.predictions
        })
        
        # Create sample metrics
        metrics = {
            'MAE': 10.5,
            'RMSE': 15.2,
            'MAPE': 2.3,
            'R2': 0.95,
            'Directional_Accuracy': 75.5
        }
        
        report = self.viz_manager.create_prediction_report(
            predictions=predictions_df,
            metrics=metrics,
            report_name='test_report',
            save_html=False,
            save_json=True
        )
        
        # Verify report structure
        self.assertIsNotNone(report)
        self.assertIsInstance(report, dict)
        self.assertIn('report_name', report)
        self.assertIn('generated_at', report)
        self.assertIn('metrics', report)
        self.assertIn('predictions', report)
        
        # Verify metrics are included
        self.assertEqual(report['metrics'], metrics)
        
        # Verify predictions count
        self.assertEqual(report['predictions']['count'], len(predictions_df))
    
    def test_create_prediction_report_with_plots(self):
        """Test prediction report creation with plots."""
        # Requirements: 9.5
        
        # Create sample predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': self.dates,
            'Predicted': self.predictions
        })
        
        metrics = {'MAE': 10.5, 'RMSE': 15.2}
        
        # Create sample plots
        fig1 = self.viz_manager.plot_time_series_with_predictions(
            actual=self.actual_prices,
            predictions=self.predictions,
            show=False
        )
        
        fig2 = self.viz_manager.plot_training_history(
            history=self.training_history,
            show=False
        )
        
        plots = [fig1, fig2]
        
        report = self.viz_manager.create_prediction_report(
            predictions=predictions_df,
            metrics=metrics,
            plots=plots,
            report_name='test_report_with_plots',
            save_html=False,
            save_json=False
        )
        
        # Verify plots are saved
        self.assertIn('plot_paths', report)
        self.assertEqual(len(report['plot_paths']), 2)
        
        # Verify plot files exist
        for plot_path in report['plot_paths']:
            self.assertTrue(Path(plot_path).exists())
    
    def test_create_prediction_report_json_output(self):
        """Test prediction report JSON output."""
        # Requirements: 9.5
        
        predictions_df = pd.DataFrame({
            'Date': self.dates,
            'Predicted': self.predictions
        })
        
        metrics = {'MAE': 10.5}
        
        report = self.viz_manager.create_prediction_report(
            predictions=predictions_df,
            metrics=metrics,
            report_name='test_json_report',
            save_html=False,
            save_json=True
        )
        
        # Verify JSON file is created
        self.assertIn('json_path', report)
        json_path = Path(report['json_path'])
        self.assertTrue(json_path.exists())
        
        # Verify JSON content is valid
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        self.assertIn('report_name', json_data)
        self.assertIn('metrics', json_data)
        self.assertEqual(json_data['report_name'], 'test_json_report')
    
    def test_create_prediction_report_html_output(self):
        """Test prediction report HTML output."""
        # Requirements: 9.5
        
        predictions_df = pd.DataFrame({
            'Date': self.dates,
            'Predicted': self.predictions
        })
        
        metrics = {
            'MAE': 10.5,
            'RMSE': 15.2,
            'MAPE': 2.3
        }
        
        report = self.viz_manager.create_prediction_report(
            predictions=predictions_df,
            metrics=metrics,
            report_name='test_html_report',
            save_html=True,
            save_json=False
        )
        
        # Verify HTML file is created
        self.assertIn('html_path', report)
        html_path = Path(report['html_path'])
        self.assertTrue(html_path.exists())
        
        # Verify HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        self.assertIn('<html>', html_content)
        self.assertIn('Gold Price Prediction Report', html_content)
        self.assertIn('MAE', html_content)
        self.assertIn('10.5', html_content)
    
    def test_create_prediction_report_csv_export(self):
        """Test prediction report exports predictions to CSV."""
        # Requirements: 9.5
        
        predictions_df = pd.DataFrame({
            'Date': self.dates,
            'Predicted': self.predictions
        })
        
        metrics = {'MAE': 10.5}
        
        report = self.viz_manager.create_prediction_report(
            predictions=predictions_df,
            metrics=metrics,
            report_name='test_csv_report',
            save_html=False,
            save_json=False
        )
        
        # Verify CSV file is created
        self.assertIn('predictions_csv_path', report)
        csv_path = Path(report['predictions_csv_path'])
        self.assertTrue(csv_path.exists())
        
        # Verify CSV content
        loaded_df = pd.read_csv(csv_path)
        self.assertEqual(len(loaded_df), len(predictions_df))
    
    def test_create_prediction_report_dict_input(self):
        """Test prediction report with dictionary input."""
        # Requirements: 9.5
        
        predictions_dict = {
            'count': 50,
            'mean_prediction': 1850.5
        }
        
        metrics = {'MAE': 10.5}
        
        report = self.viz_manager.create_prediction_report(
            predictions=predictions_dict,
            metrics=metrics,
            report_name='test_dict_report',
            save_html=False,
            save_json=False
        )
        
        # Verify report is created
        self.assertIsNotNone(report)
        self.assertEqual(report['predictions'], predictions_dict)
    
    def test_integration_complete_visualization_workflow(self):
        """Integration test for complete visualization workflow."""
        # Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
        
        # Create visualization manager
        viz = VisualizationManager()
        
        # Step 1: Plot time series with predictions and confidence intervals
        fig1 = viz.plot_time_series_with_predictions(
            actual=self.actual_prices,
            predictions=self.predictions,
            confidence_intervals=self.confidence_intervals,
            dates=self.dates,
            title="Integration Test: Gold Price Predictions",
            save_path=self.test_dir / "integration_time_series.png",
            show=False
        )
        self.assertIsNotNone(fig1)
        self.assertTrue((self.test_dir / "integration_time_series.png").exists())
        
        # Step 2: Plot feature importance
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.random.rand(10)
        
        mock_model = MockModel()
        feature_names = [f'Feature_{i}' for i in range(10)]
        
        fig2 = viz.plot_feature_importance(
            model=mock_model,
            feature_names=feature_names,
            save_path=self.test_dir / "integration_feature_importance.png",
            show=False
        )
        self.assertIsNotNone(fig2)
        self.assertTrue((self.test_dir / "integration_feature_importance.png").exists())
        
        # Step 3: Plot indicators overlay
        gold_df = pd.DataFrame({
            'Date': self.dates,
            'Close': self.actual_prices
        })
        
        indicators = {
            'DXY': pd.DataFrame({
                'Date': self.dates,
                'Value': 100 + np.random.randn(self.n_samples)
            }),
            'Oil': pd.DataFrame({
                'Date': self.dates,
                'Value': 80 + np.random.randn(self.n_samples) * 5
            })
        }
        
        fig3 = viz.plot_indicators_overlay(
            gold_prices=gold_df,
            indicators=indicators,
            save_path=self.test_dir / "integration_indicators.png",
            show=False
        )
        self.assertIsNotNone(fig3)
        self.assertTrue((self.test_dir / "integration_indicators.png").exists())
        
        # Step 4: Plot training history
        fig4 = viz.plot_training_history(
            history=self.training_history,
            save_path=self.test_dir / "integration_training_history.png",
            show=False
        )
        self.assertIsNotNone(fig4)
        self.assertTrue((self.test_dir / "integration_training_history.png").exists())
        
        # Step 5: Create comprehensive report
        predictions_df = pd.DataFrame({
            'Date': self.dates,
            'Actual': self.actual_prices,
            'Predicted': self.predictions,
            'Lower_CI': self.confidence_intervals[:, 0],
            'Upper_CI': self.confidence_intervals[:, 1]
        })
        
        metrics = {
            'MAE': 12.5,
            'RMSE': 18.3,
            'MAPE': 2.1,
            'R2': 0.94,
            'Directional_Accuracy': 78.2
        }
        
        report = viz.create_prediction_report(
            predictions=predictions_df,
            metrics=metrics,
            plots=[fig1, fig2, fig3, fig4],
            report_name='integration_test_report',
            save_html=True,
            save_json=True
        )
        
        # Verify report is complete
        self.assertIsNotNone(report)
        self.assertIn('json_path', report)
        self.assertIn('html_path', report)
        self.assertIn('predictions_csv_path', report)
        self.assertEqual(len(report['plot_paths']), 4)
        
        # Verify all files exist
        self.assertTrue(Path(report['json_path']).exists())
        self.assertTrue(Path(report['html_path']).exists())
        self.assertTrue(Path(report['predictions_csv_path']).exists())
        
        for plot_path in report['plot_paths']:
            self.assertTrue(Path(plot_path).exists())
        
        # Verify report contains all metrics
        self.assertEqual(report['metrics'], metrics)
        
        # Log success
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUCCESSFUL")
        print("=" * 60)
        print(f"All visualization functions working correctly")
        print(f"Report generated: {report['report_name']}")
        print(f"Total files created: {4 + 1 + 1 + 1 + len(report['plot_paths'])}")
        print("=" * 60)
    
    def test_integration_error_handling(self):
        """Integration test for error handling across visualization methods."""
        # Requirements: 9.1, 9.3, 9.5
        
        viz = VisualizationManager()
        
        # Test invalid model for feature importance
        class InvalidModel:
            pass
        
        with self.assertRaises(AttributeError):
            viz.plot_feature_importance(
                model=InvalidModel(),
                feature_names=['f1', 'f2'],
                show=False
            )
        
        # Test invalid training history
        with self.assertRaises(ValueError):
            viz.plot_training_history(
                history={'invalid_key': [1, 2, 3]},
                show=False
            )
        
        # Test empty predictions
        empty_predictions = pd.DataFrame()
        metrics = {}
        
        # Should not raise error, but handle gracefully
        report = viz.create_prediction_report(
            predictions=empty_predictions,
            metrics=metrics,
            report_name='test_empty_report',
            save_html=False,
            save_json=False
        )
        
        self.assertIsNotNone(report)


if __name__ == '__main__':
    unittest.main()
