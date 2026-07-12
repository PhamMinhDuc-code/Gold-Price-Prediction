"""
Model Evaluation Module

This module implements the ModelEvaluator class for evaluating model performance
with comprehensive metrics and visualizations.

Classes:
    - ModelEvaluator: Main class for model evaluation and reporting

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import logging

from config import Config


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

# Add handler if not already present
if not logger.handlers:
    log_file = Config.get_log_path('src.model_evaluator.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class ModelEvaluator:
    """
    Model Evaluator for comprehensive performance assessment.
    
    This class provides functionality to:
    - Calculate regression metrics (MAE, RMSE, MAPE, R²)
    - Calculate directional accuracy
    - Generate residual plots
    - Generate prediction vs actual plots
    - Create comprehensive performance reports
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
    """
    
    def __init__(self, y_true: Optional[np.ndarray] = None, 
                 y_pred: Optional[np.ndarray] = None,
                 dates: Optional[List[datetime]] = None):
        """
        Initialize ModelEvaluator with test data and predictions.
        
        Args:
            y_true: Actual values (optional, can be set later)
            y_pred: Predicted values (optional, can be set later)
            dates: Corresponding dates for time series (optional)
        
        Requirements: 7.1
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.dates = dates
        
        # Store computed metrics
        self.metrics = {}
        
        # Store figure objects
        self.figures = {}
        
        logger.info("ModelEvaluator initialized")
        
        if y_true is not None and y_pred is not None:
            self._validate_inputs()
    
    def set_data(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 dates: Optional[List[datetime]] = None):
        """
        Set or update test data and predictions.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Corresponding dates for time series (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.dates = dates
        
        self._validate_inputs()
        
        # Clear previous metrics and figures
        self.metrics = {}
        self.figures = {}
        
        logger.info(f"Data set: {len(y_true)} samples")
    
    def _validate_inputs(self):
        """Validate that inputs have compatible shapes."""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Both y_true and y_pred must be provided")
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"Shape mismatch: y_true has {len(self.y_true)} samples, "
                f"y_pred has {len(self.y_pred)} samples"
            )
        
        if self.dates is not None and len(self.dates) != len(self.y_true):
            raise ValueError(
                f"Shape mismatch: dates has {len(self.dates)} entries, "
                f"but data has {len(self.y_true)} samples"
            )
        
        logger.debug("Input validation passed")
    
    def calculate_mae(self, y_true: Optional[np.ndarray] = None,
                     y_pred: Optional[np.ndarray] = None) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        MAE = (1/n) * Σ|y_true - y_pred|
        
        Args:
            y_true: Actual values (uses instance data if None)
            y_pred: Predicted values (uses instance data if None)
        
        Returns:
            MAE value
        
        Requirements: 7.1
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided")
        
        mae = np.mean(np.abs(y_true - y_pred))
        
        self.metrics['MAE'] = float(mae)
        logger.info(f"MAE: {mae:.4f}")
        
        return mae
    
    def calculate_rmse(self, y_true: Optional[np.ndarray] = None,
                      y_pred: Optional[np.ndarray] = None) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
        
        Args:
            y_true: Actual values (uses instance data if None)
            y_pred: Predicted values (uses instance data if None)
        
        Returns:
            RMSE value
        
        Requirements: 7.2
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided")
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        self.metrics['RMSE'] = float(rmse)
        logger.info(f"RMSE: {rmse:.4f}")
        
        return rmse
    
    def calculate_mape(self, y_true: Optional[np.ndarray] = None,
                      y_pred: Optional[np.ndarray] = None,
                      epsilon: float = 1e-10) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        MAPE = (100/n) * Σ|(y_true - y_pred) / y_true|
        
        Args:
            y_true: Actual values (uses instance data if None)
            y_pred: Predicted values (uses instance data if None)
            epsilon: Small value to avoid division by zero
        
        Returns:
            MAPE value (percentage)
        
        Requirements: 7.3
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided")
        
        # Avoid division by zero
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        
        self.metrics['MAPE'] = float(mape)
        logger.info(f"MAPE: {mape:.4f}%")
        
        return mape
    
    def calculate_r2(self, y_true: Optional[np.ndarray] = None,
                    y_pred: Optional[np.ndarray] = None) -> float:
        """
        Calculate coefficient of determination (R²).
        
        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - mean(y_true))²
        
        Args:
            y_true: Actual values (uses instance data if None)
            y_pred: Predicted values (uses instance data if None)
        
        Returns:
            R² value (ranges from -∞ to 1, where 1 is perfect)
        
        Requirements: 7.4
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided")
        
        # Calculate sum of squared residuals
        ss_res = np.sum((y_true - y_pred) ** 2)
        
        # Calculate total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Calculate R²
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        self.metrics['R2'] = float(r2)
        logger.info(f"R²: {r2:.4f}")
        
        return r2
    
    def calculate_directional_accuracy(self, y_true: Optional[np.ndarray] = None,
                                      y_pred: Optional[np.ndarray] = None) -> float:
        """
        Calculate directional accuracy (percentage of correct price movement predictions).
        
        Measures the percentage of times the model correctly predicted whether
        the price would go up or down compared to the previous value.
        
        Args:
            y_true: Actual values (uses instance data if None)
            y_pred: Predicted values (uses instance data if None)
        
        Returns:
            Directional accuracy as percentage (0-100)
        
        Requirements: 7.6
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided")
        
        if len(y_true) < 2:
            logger.warning("Need at least 2 samples to calculate directional accuracy")
            return 0.0
        
        # Calculate direction changes
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Count correct predictions
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        
        directional_accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        self.metrics['Directional_Accuracy'] = float(directional_accuracy)
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        return directional_accuracy
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all regression metrics at once.
        
        Returns:
            Dictionary containing all metrics
        
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.6
        """
        logger.info("Calculating all evaluation metrics")
        
        self.calculate_mae()
        self.calculate_rmse()
        self.calculate_mape()
        self.calculate_r2()
        self.calculate_directional_accuracy()
        
        logger.info("All metrics calculated successfully")
        logger.info(f"Metrics: {self.metrics}")
        
        return self.metrics.copy()
    
    def plot_residuals(self, y_true: Optional[np.ndarray] = None,
                      y_pred: Optional[np.ndarray] = None,
                      dates: Optional[List[datetime]] = None,
                      save_path: Optional[Path] = None,
                      show: bool = False) -> plt.Figure:
        """
        Generate residual plots showing prediction errors over time.
        
        Creates a plot showing residuals (prediction errors) over time,
        helping identify patterns in model errors.
        
        Args:
            y_true: Actual values (uses instance data if None)
            y_pred: Predicted values (uses instance data if None)
            dates: Corresponding dates (uses instance dates if None)
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        
        Returns:
            Matplotlib Figure object
        
        Requirements: 7.5
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        if dates is None:
            dates = self.dates
        
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided")
        
        logger.info("Generating residual plot")
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Use dates if available, otherwise use indices
        x_values = dates if dates is not None else np.arange(len(residuals))
        
        # Plot 1: Residuals over time
        axes[0].plot(x_values, residuals, marker='o', linestyle='-', alpha=0.7, markersize=3)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Time' if dates is not None else 'Sample Index')
        axes[0].set_ylabel('Residual (Actual - Predicted)')
        axes[0].set_title('Residuals Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if dates is not None:
            axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[1].set_xlabel('Residual Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        # Store figure
        self.figures['residuals'] = fig
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info("Residual plot generated successfully")
        
        return fig
    
    def plot_predictions_vs_actual(self, y_true: Optional[np.ndarray] = None,
                                   y_pred: Optional[np.ndarray] = None,
                                   dates: Optional[List[datetime]] = None,
                                   save_path: Optional[Path] = None,
                                   show: bool = False) -> plt.Figure:
        """
        Plot predictions vs actual values over time.
        
        Creates a time series plot comparing predicted and actual values,
        helping visualize model performance.
        
        Args:
            y_true: Actual values (uses instance data if None)
            y_pred: Predicted values (uses instance data if None)
            dates: Corresponding dates (uses instance dates if None)
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        
        Returns:
            Matplotlib Figure object
        
        Requirements: 7.5
        """
        if y_true is None:
            y_true = self.y_true
        if y_pred is None:
            y_pred = self.y_pred
        if dates is None:
            dates = self.dates
        
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided")
        
        logger.info("Generating predictions vs actual plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Use dates if available, otherwise use indices
        x_values = dates if dates is not None else np.arange(len(y_true))
        
        # Plot actual and predicted values
        ax.plot(x_values, y_true, label='Actual', color='blue', linewidth=2, alpha=0.7)
        ax.plot(x_values, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Time' if dates is not None else 'Sample Index')
        ax.set_ylabel('Gold Price')
        ax.set_title('Predictions vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if dates is not None:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        # Store figure
        self.figures['predictions_vs_actual'] = fig
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info("Predictions vs actual plot generated successfully")
        
        return fig
    
    def generate_performance_report(self, report_name: Optional[str] = None,
                                   save_json: bool = True,
                                   save_plots: bool = True) -> Dict:
        """
        Generate comprehensive performance report.
        
        Compiles all metrics and plots into a comprehensive report and saves
        as JSON. Optionally saves plots as PNG files.
        
        Args:
            report_name: Base name for report files (default: timestamp-based)
            save_json: Whether to save report as JSON
            save_plots: Whether to save plots as PNG files
        
        Returns:
            Dictionary containing complete performance report
        
        Requirements: 7.7
        """
        logger.info("Generating comprehensive performance report")
        
        # Calculate all metrics if not already done
        if not self.metrics:
            self.calculate_all_metrics()
        
        # Generate timestamp for report name if not provided
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"performance_report_{timestamp}"
        
        # Create report dictionary
        report = {
            'report_name': report_name,
            'generated_at': datetime.now().isoformat(),
            'sample_count': len(self.y_true) if self.y_true is not None else 0,
            'metrics': self.metrics.copy(),
            'summary': {
                'mae': self.metrics.get('MAE'),
                'rmse': self.metrics.get('RMSE'),
                'mape': self.metrics.get('MAPE'),
                'r2': self.metrics.get('R2'),
                'directional_accuracy': self.metrics.get('Directional_Accuracy')
            },
            'date_range': None
        }
        
        # Add date range if available
        if self.dates is not None:
            report['date_range'] = {
                'start': self.dates[0].isoformat() if isinstance(self.dates[0], datetime) else str(self.dates[0]),
                'end': self.dates[-1].isoformat() if isinstance(self.dates[-1], datetime) else str(self.dates[-1])
            }
        
        # Save JSON report
        if save_json:
            json_path = Config.REPORTS_DIR / f"{report_name}.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"JSON report saved to {json_path}")
            report['json_path'] = str(json_path)
        
        # Generate and save plots
        if save_plots:
            # Generate residual plot
            residual_path = Config.REPORTS_DIR / f"{report_name}_residuals.png"
            self.plot_residuals(save_path=residual_path, show=False)
            report['residual_plot_path'] = str(residual_path)
            
            # Generate predictions vs actual plot
            pred_plot_path = Config.REPORTS_DIR / f"{report_name}_predictions.png"
            self.plot_predictions_vs_actual(save_path=pred_plot_path, show=False)
            report['predictions_plot_path'] = str(pred_plot_path)
            
            logger.info("All plots saved successfully")
        
        # Log summary
        logger.info("=" * 60)
        logger.info("PERFORMANCE REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Sample Count: {report['sample_count']}")
        logger.info(f"MAE: {report['summary']['mae']:.4f}")
        logger.info(f"RMSE: {report['summary']['rmse']:.4f}")
        logger.info(f"MAPE: {report['summary']['mape']:.2f}%")
        logger.info(f"R²: {report['summary']['r2']:.4f}")
        logger.info(f"Directional Accuracy: {report['summary']['directional_accuracy']:.2f}%")
        logger.info("=" * 60)
        
        logger.info("Performance report generated successfully")
        
        return report


if __name__ == "__main__":
    # Example usage
    print("ModelEvaluator module loaded successfully")
    
    # Create sample data for demonstration
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 100
    y_pred = y_true + np.random.randn(100) * 2  # Add some noise
    
    # Initialize evaluator
    evaluator = ModelEvaluator(y_true, y_pred)
    
    print("\nModelEvaluator initialized with sample data")
    print("Available methods:")
    print("  - calculate_mae()")
    print("  - calculate_rmse()")
    print("  - calculate_mape()")
    print("  - calculate_r2()")
    print("  - calculate_directional_accuracy()")
    print("  - calculate_all_metrics()")
    print("  - plot_residuals()")
    print("  - plot_predictions_vs_actual()")
    print("  - generate_performance_report()")
    
    # Calculate all metrics
    metrics = evaluator.calculate_all_metrics()
    print("\nCalculated Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
