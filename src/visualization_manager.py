"""
Visualization and Reporting Module

This module implements the VisualizationManager class for generating comprehensive
visualizations and prediction reports for gold price predictions.

Classes:
    - VisualizationManager: Main class for visualization and reporting

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import logging
from matplotlib.figure import Figure

from config import Config


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

# Add handler if not already present
if not logger.handlers:
    log_file = Config.get_log_path('src.visualization_manager.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class VisualizationManager:
    """
    Visualization Manager for generating comprehensive visualizations.
    
    This class provides functionality to:
    - Plot time series with predictions and confidence intervals
    - Plot feature importance for tree-based models
    - Plot economic indicators overlay with gold prices
    - Plot training history (loss curves)
    - Create comprehensive prediction reports
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    
    def __init__(self):
        """
        Initialize VisualizationManager with matplotlib/seaborn configuration.
        
        Sets up consistent styling for all plots including color schemes,
        figure sizes, and plot styles.
        
        Requirements: 9.1
        """
        # Set up matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
        
        # Set seaborn style
        sns.set_palette("husl")
        
        # Configure default figure parameters
        plt.rcParams['figure.figsize'] = Config.FIGURE_SIZE
        plt.rcParams['figure.dpi'] = Config.FIGURE_DPI
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['font.size'] = 10
        
        # Store figure objects
        self.figures = {}
        
        logger.info("VisualizationManager initialized with matplotlib/seaborn configuration")
        logger.info(f"Default figure size: {Config.FIGURE_SIZE}, DPI: {Config.FIGURE_DPI}")
    
    def plot_time_series_with_predictions(self, 
                                          actual: Union[pd.DataFrame, np.ndarray],
                                          predictions: Union[pd.DataFrame, np.ndarray],
                                          confidence_intervals: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                                          dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
                                          title: str = "Gold Price Predictions with Confidence Intervals",
                                          save_path: Optional[Path] = None,
                                          show: bool = False) -> Figure:
        """
        Plot time series with predictions and confidence bands.
        
        Creates a comprehensive time series plot showing actual prices,
        predictions, and optional confidence intervals with different colors
        and styles for clarity.
        
        Args:
            actual: Actual gold prices (DataFrame or array)
            predictions: Predicted gold prices (DataFrame or array)
            confidence_intervals: Optional confidence intervals (DataFrame or array with shape (n, 2))
            dates: Optional dates for x-axis
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        
        Returns:
            Matplotlib Figure object
        
        Requirements: 9.1, 9.2
        """
        logger.info("Generating time series prediction plot with confidence intervals")
        
        # Convert inputs to arrays if needed
        if isinstance(actual, pd.DataFrame):
            actual = actual.values.flatten()
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values.flatten()
        if isinstance(confidence_intervals, pd.DataFrame):
            confidence_intervals = confidence_intervals.values
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Use dates if available, otherwise use indices
        if dates is None:
            x_values = np.arange(len(actual))
            xlabel = 'Sample Index'
        else:
            x_values = dates
            xlabel = 'Date'
        
        # Plot actual prices
        ax.plot(x_values[:len(actual)], actual, 
                label='Actual Price', color='blue', linewidth=2, alpha=0.8)
        
        # Plot predictions
        pred_x = x_values[len(actual)-len(predictions):len(actual)] if len(predictions) < len(actual) else x_values[:len(predictions)]
        ax.plot(pred_x, predictions,
                label='Predicted Price', color='red', linewidth=2, alpha=0.8, linestyle='--')
        
        # Plot confidence intervals if provided
        if confidence_intervals is not None:
            lower_bound = confidence_intervals[:, 0]
            upper_bound = confidence_intervals[:, 1]
            
            ax.fill_between(pred_x, lower_bound, upper_bound,
                           alpha=0.2, color='red', label='95% Confidence Interval')
        
        # Configure plot
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Gold Price (USD)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if using dates
        if dates is not None:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        # Store figure
        self.figures['time_series_predictions'] = fig
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info("Time series prediction plot generated successfully")
        
        return fig
    
    def plot_feature_importance(self,
                               model: Any,
                               feature_names: List[str],
                               top_n: int = 20,
                               title: str = "Feature Importance",
                               save_path: Optional[Path] = None,
                               show: bool = False) -> Figure:
        """
        Generate feature importance plot for tree-based models.
        
        Creates a horizontal bar chart showing which features contribute most
        to the model's predictions. Only applicable for tree-based models
        (XGBoost, Random Forest).
        
        Args:
            model: Trained model (must have feature_importances_ attribute)
            feature_names: List of feature names
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        
        Returns:
            Matplotlib Figure object
        
        Requirements: 9.3
        """
        logger.info(f"Generating feature importance plot for top {top_n} features")
        
        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_'):
            logger.error("Model does not have feature_importances_ attribute")
            raise AttributeError("Model must be a tree-based model with feature_importances_ attribute")
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Ensure feature_names length matches importances
        if len(feature_names) != len(importances):
            logger.warning(f"Feature names length ({len(feature_names)}) != importances length ({len(importances)})")
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                       color='steelblue', alpha=0.8)
        
        # Color the bars by importance (gradient)
        colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Configure plot
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to show most important at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        # Store figure
        self.figures['feature_importance'] = fig
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info(f"Feature importance plot generated successfully with {len(importance_df)} features")
        
        return fig
    
    def plot_indicators_overlay(self,
                               gold_prices: pd.DataFrame,
                               indicators: Dict[str, pd.DataFrame],
                               date_column: str = 'Date',
                               price_column: str = 'Close',
                               title: str = "Gold Prices with Economic Indicators",
                               save_path: Optional[Path] = None,
                               show: bool = False) -> Figure:
        """
        Plot economic indicators overlaid with gold prices.
        
        Creates a multi-panel plot showing gold prices and economic indicators
        (DXY, Oil, Treasury yields) with secondary y-axes where appropriate.
        
        Args:
            gold_prices: DataFrame with gold price data
            indicators: Dictionary of indicator DataFrames {name: df}
            date_column: Name of date column
            price_column: Name of price column
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        
        Returns:
            Matplotlib Figure object
        
        Requirements: 9.4
        """
        logger.info(f"Generating indicators overlay plot with {len(indicators)} indicators")
        
        # Determine number of subplots (1 for gold + 1 per indicator)
        n_plots = 1 + len(indicators)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
        
        # Ensure axes is always a list
        if n_plots == 1:
            axes = [axes]
        
        # Extract dates and gold prices
        if date_column in gold_prices.columns:
            dates = pd.to_datetime(gold_prices[date_column])
        else:
            dates = gold_prices.index
        
        gold_values = gold_prices[price_column].values
        
        # Plot 1: Gold prices
        axes[0].plot(dates, gold_values, color='gold', linewidth=2, label='Gold Price')
        axes[0].set_ylabel('Gold Price (USD)', fontsize=11, fontweight='bold')
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot indicators on separate panels
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, (indicator_name, indicator_df) in enumerate(indicators.items(), start=1):
            ax = axes[idx]
            
            # Extract indicator dates and values
            if date_column in indicator_df.columns:
                ind_dates = pd.to_datetime(indicator_df[date_column])
            else:
                ind_dates = indicator_df.index
            
            # Get value column (assume first non-date column if not specified)
            value_cols = [col for col in indicator_df.columns if col != date_column]
            if value_cols:
                ind_values = indicator_df[value_cols[0]].values
            else:
                ind_values = indicator_df.iloc[:, 0].values
            
            # Plot indicator
            color = colors[idx % len(colors)]
            ax.plot(ind_dates, ind_values, color=color, linewidth=2, label=indicator_name)
            ax.set_ylabel(indicator_name, fontsize=11, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Set x-label on bottom plot
        axes[-1].set_xlabel('Date', fontsize=12)
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Indicators overlay plot saved to {save_path}")
        
        # Store figure
        self.figures['indicators_overlay'] = fig
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info("Indicators overlay plot generated successfully")
        
        return fig
    
    def plot_training_history(self,
                             history: Dict,
                             title: str = "Model Training History",
                             save_path: Optional[Path] = None,
                             show: bool = False) -> Figure:
        """
        Plot training and validation loss curves.
        
        Creates a plot showing training and validation loss over epochs,
        highlighting the best epoch with early stopping.
        
        Args:
            history: Training history dictionary with 'loss' and 'val_loss' keys
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        
        Returns:
            Matplotlib Figure object
        
        Requirements: 5.6
        """
        logger.info("Generating training history plot")
        
        # Validate history dictionary
        if 'loss' not in history:
            logger.error("History dictionary missing 'loss' key")
            raise ValueError("History dictionary must contain 'loss' key")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get epochs
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot training loss
        ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
        
        # Plot validation loss if available
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
            
            # Highlight best epoch (minimum validation loss)
            best_epoch = np.argmin(history['val_loss']) + 1
            best_val_loss = min(history['val_loss'])
            
            ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'Best Epoch ({best_epoch})')
            ax.plot(best_epoch, best_val_loss, 'g*', markersize=15, label=f'Best Val Loss: {best_val_loss:.4f}')
        
        # Configure plot
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        # Store figure
        self.figures['training_history'] = fig
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info("Training history plot generated successfully")
        
        return fig
    
    def create_prediction_report(self,
                                predictions: Union[pd.DataFrame, Dict],
                                metrics: Dict,
                                plots: Optional[List[Figure]] = None,
                                report_name: Optional[str] = None,
                                save_html: bool = True,
                                save_json: bool = True) -> Dict:
        """
        Create comprehensive prediction report.
        
        Compiles predictions, metrics, and plots into a single comprehensive
        report and exports as HTML or JSON format.
        
        Args:
            predictions: DataFrame or dict with prediction data
            metrics: Dictionary of evaluation metrics
            plots: Optional list of Figure objects to include
            report_name: Base name for report files (default: timestamp-based)
            save_html: Whether to save report as HTML
            save_json: Whether to save report as JSON
        
        Returns:
            Dictionary containing complete report data
        
        Requirements: 9.5
        """
        logger.info("Creating comprehensive prediction report")
        
        # Generate timestamp for report name if not provided
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"prediction_report_{timestamp}"
        
        # Build report dictionary
        report = {
            'report_name': report_name,
            'generated_at': datetime.now().isoformat(),
            'metrics': metrics.copy() if metrics else {},
            'predictions': {},
            'plot_paths': []
        }
        
        # Add prediction data
        if isinstance(predictions, pd.DataFrame):
            report['predictions'] = {
                'count': len(predictions),
                'columns': list(predictions.columns),
                'data': predictions.to_dict('records')[:100]  # Limit to first 100 for JSON size
            }
            
            # Save full predictions as CSV
            csv_path = Config.REPORTS_DIR / f"{report_name}_predictions.csv"
            predictions.to_csv(csv_path, index=False)
            report['predictions_csv_path'] = str(csv_path)
            logger.info(f"Predictions saved to CSV: {csv_path}")
        elif isinstance(predictions, dict):
            report['predictions'] = predictions
        
        # Save plots as images
        if plots:
            for idx, fig in enumerate(plots):
                plot_path = Config.REPORTS_DIR / f"{report_name}_plot_{idx+1}.png"
                fig.savefig(plot_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
                report['plot_paths'].append(str(plot_path))
                logger.info(f"Plot {idx+1} saved to {plot_path}")
        
        # Save JSON report
        if save_json:
            # Create a JSON-serializable version (exclude plot data)
            json_report = {
                'report_name': report['report_name'],
                'generated_at': report['generated_at'],
                'metrics': report['metrics'],
                'predictions_summary': {
                    'count': report['predictions'].get('count', 0),
                    'columns': report['predictions'].get('columns', [])
                },
                'plot_paths': report['plot_paths']
            }
            
            if 'predictions_csv_path' in report:
                json_report['predictions_csv_path'] = report['predictions_csv_path']
            
            json_path = Config.REPORTS_DIR / f"{report_name}.json"
            with open(json_path, 'w') as f:
                json.dump(json_report, f, indent=4)
            logger.info(f"JSON report saved to {json_path}")
            report['json_path'] = str(json_path)
        
        # Save HTML report
        if save_html:
            html_path = Config.REPORTS_DIR / f"{report_name}.html"
            self._generate_html_report(report, html_path)
            report['html_path'] = str(html_path)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("PREDICTION REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Report Name: {report_name}")
        logger.info(f"Metrics: {len(report['metrics'])} metrics included")
        logger.info(f"Predictions: {report['predictions'].get('count', 0)} records")
        logger.info(f"Plots: {len(report['plot_paths'])} plots included")
        if 'json_path' in report:
            logger.info(f"JSON Report: {report['json_path']}")
        if 'html_path' in report:
            logger.info(f"HTML Report: {report['html_path']}")
        logger.info("=" * 60)
        
        logger.info("Comprehensive prediction report created successfully")
        
        return report
    
    def _generate_html_report(self, report: Dict, html_path: Path) -> None:
        """
        Generate HTML report from report dictionary.
        
        Args:
            report: Report dictionary
            html_path: Path to save HTML file
        """
        logger.info(f"Generating HTML report at {html_path}")
        
        # Build HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report['report_name']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 4px solid #4CAF50;
        }}
        .metric-name {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.5em;
            color: #333;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #999;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Gold Price Prediction Report</h1>
        <p><strong>Report Name:</strong> {report['report_name']}</p>
        <p><strong>Generated:</strong> {report['generated_at']}</p>
"""
        
        # Add metrics section
        if report['metrics']:
            html_content += """
        <h2>Performance Metrics</h2>
        <div class="metrics-container">
"""
            for metric_name, metric_value in report['metrics'].items():
                if isinstance(metric_value, float):
                    formatted_value = f"{metric_value:.4f}"
                else:
                    formatted_value = str(metric_value)
                
                html_content += f"""
            <div class="metric">
                <div class="metric-name">{metric_name}</div>
                <div class="metric-value">{formatted_value}</div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Add predictions summary
        if 'predictions' in report and report['predictions']:
            pred_count = report['predictions'].get('count', 0)
            html_content += f"""
        <h2>Predictions Summary</h2>
        <p><strong>Total Predictions:</strong> {pred_count}</p>
"""
            if 'predictions_csv_path' in report:
                csv_filename = Path(report['predictions_csv_path']).name
                html_content += f"""
        <p><strong>Full predictions available in:</strong> {csv_filename}</p>
"""
        
        # Add plots
        if report['plot_paths']:
            html_content += """
        <h2>Visualizations</h2>
"""
            for idx, plot_path in enumerate(report['plot_paths']):
                plot_filename = Path(plot_path).name
                html_content += f"""
        <div class="plot">
            <h3>Plot {idx+1}</h3>
            <img src="{plot_filename}" alt="Plot {idx+1}">
        </div>
"""
        
        # Close HTML
        html_content += """
        <div class="footer">
            <p>Generated by Gold Price Prediction System</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated successfully at {html_path}")


if __name__ == "__main__":
    # Example usage
    print("VisualizationManager module loaded successfully")
    
    # Initialize visualization manager
    viz_manager = VisualizationManager()
    
    print("\nVisualizationManager initialized")
    print("Available methods:")
    print("  - plot_time_series_with_predictions()")
    print("  - plot_feature_importance()")
    print("  - plot_indicators_overlay()")
    print("  - plot_training_history()")
    print("  - create_prediction_report()")
