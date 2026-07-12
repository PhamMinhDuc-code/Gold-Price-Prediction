"""
Comprehensive Demo Script for Visualization Manager

This script demonstrates all functionality of the VisualizationManager class:
1. Time series prediction plotting with confidence intervals
2. Feature importance plotting for tree-based models
3. Economic indicators overlay plotting
4. Training history plotting
5. Comprehensive prediction report generation (HTML/PDF/JSON)

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import project modules
from src.visualization_manager import VisualizationManager
from config import Config

print("=" * 80)
print("VISUALIZATION MANAGER COMPREHENSIVE DEMO")
print("=" * 80)
print()

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STEP 1: Prepare Sample Data
# ============================================================================
print("Step 1: Preparing sample data...")

# Create sample dates (100 days)
n_samples = 100
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')

# Create sample actual gold prices (random walk)
actual_prices = 1800 + np.cumsum(np.random.randn(n_samples) * 5)

# Create sample predictions (actual + noise)
predictions = actual_prices + np.random.randn(n_samples) * 10

# Create confidence intervals (95%)
std_error = 20
confidence_intervals = np.column_stack([
    predictions - 1.96 * std_error,  # Lower bound
    predictions + 1.96 * std_error   # Upper bound
])

# Create sample feature names and importances
feature_names = [
    'Close_lag_1', 'Close_lag_7', 'Close_lag_14', 'Close_lag_30',
    'Close_ma_7', 'Close_ma_14', 'Close_ma_30', 'Close_ma_90',
    'RSI_14', 'MACD', 'MACD_signal', 'BB_upper',
    'DXY', 'Oil', 'Treasury_10Y',
    'Gold_Oil_ratio', 'Gold_DXY_corr',
    'day_of_week', 'month', 'quarter'
]
feature_importances = np.random.rand(len(feature_names))
feature_importances = feature_importances / feature_importances.sum()  # Normalize

# Create sample economic indicators
gold_prices_df = pd.DataFrame({
    'Date': dates,
    'Close': actual_prices
})

dxy_indicator = pd.DataFrame({
    'Date': dates,
    'DXY': 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
})

oil_indicator = pd.DataFrame({
    'Date': dates,
    'Oil': 80 + np.cumsum(np.random.randn(n_samples) * 2)
})

treasury_indicator = pd.DataFrame({
    'Date': dates,
    'Treasury_10Y': 4.0 + np.cumsum(np.random.randn(n_samples) * 0.1)
})

indicators = {
    'US Dollar Index (DXY)': dxy_indicator,
    'Crude Oil': oil_indicator,
    '10Y Treasury Yield': treasury_indicator
}

# Create sample training history
training_history = {
    'loss': [0.05, 0.04, 0.035, 0.032, 0.030, 0.029, 0.028, 0.027, 0.027, 0.027, 0.028, 0.028],
    'val_loss': [0.055, 0.045, 0.038, 0.035, 0.033, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038]
}

print(f"  [OK] Generated {n_samples} days of sample data")
print(f"  [OK] Created {len(feature_names)} features with importance values")
print(f"  [OK] Created {len(indicators)} economic indicators")
print(f"  [OK] Created training history with {len(training_history['loss'])} epochs")
print()

# ============================================================================
# STEP 2: Initialize Visualization Manager
# ============================================================================
print("Step 2: Initializing VisualizationManager...")

viz_manager = VisualizationManager()

print("  [OK] VisualizationManager initialized with matplotlib/seaborn configuration")
print()

# ============================================================================
# STEP 3: Plot Time Series with Predictions and Confidence Intervals
# ============================================================================
print("Step 3: Generating time series prediction plot with confidence intervals...")
print("  Requirements: 9.1, 9.2")

fig1 = viz_manager.plot_time_series_with_predictions(
    actual=actual_prices,
    predictions=predictions,
    confidence_intervals=confidence_intervals,
    dates=dates,
    title="Gold Price Predictions with 95% Confidence Intervals",
    save_path=Config.REPORTS_DIR / "demo_viz_time_series.png",
    show=False
)

print(f"  [OK] Time series plot saved to: {Config.REPORTS_DIR / 'demo_viz_time_series.png'}")
print()

# ============================================================================
# STEP 4: Plot Feature Importance
# ============================================================================
print("Step 4: Generating feature importance plot...")
print("  Requirements: 9.3")

# Create mock model with feature importances
class MockTreeModel:
    """Mock tree-based model for demonstration."""
    def __init__(self, importances):
        self.feature_importances_ = importances

mock_model = MockTreeModel(feature_importances)

fig2 = viz_manager.plot_feature_importance(
    model=mock_model,
    feature_names=feature_names,
    top_n=15,
    title="Top 15 Most Important Features for Gold Price Prediction",
    save_path=Config.REPORTS_DIR / "demo_viz_feature_importance.png",
    show=False
)

print(f"  [OK] Feature importance plot saved to: {Config.REPORTS_DIR / 'demo_viz_feature_importance.png'}")
print()

# ============================================================================
# STEP 5: Plot Economic Indicators Overlay
# ============================================================================
print("Step 5: Generating economic indicators overlay plot...")
print("  Requirements: 9.4")

fig3 = viz_manager.plot_indicators_overlay(
    gold_prices=gold_prices_df,
    indicators=indicators,
    date_column='Date',
    price_column='Close',
    title="Gold Prices with Economic Indicators (Multi-Panel View)",
    save_path=Config.REPORTS_DIR / "demo_viz_indicators_overlay.png",
    show=False
)

print(f"  [OK] Indicators overlay plot saved to: {Config.REPORTS_DIR / 'demo_viz_indicators_overlay.png'}")
print()

# ============================================================================
# STEP 6: Plot Training History
# ============================================================================
print("Step 6: Generating training history plot...")
print("  Requirements: 5.6")

fig4 = viz_manager.plot_training_history(
    history=training_history,
    title="LSTM Model Training History - Loss Curves with Best Epoch",
    save_path=Config.REPORTS_DIR / "demo_viz_training_history.png",
    show=False
)

print(f"  [OK] Training history plot saved to: {Config.REPORTS_DIR / 'demo_viz_training_history.png'}")
print()

# ============================================================================
# STEP 7: Create Comprehensive Prediction Report
# ============================================================================
print("Step 7: Creating comprehensive prediction report (HTML + JSON)...")
print("  Requirements: 9.5")

# Prepare predictions DataFrame
predictions_df = pd.DataFrame({
    'Date': dates,
    'Actual_Price': actual_prices,
    'Predicted_Price': predictions,
    'Lower_CI_95': confidence_intervals[:, 0],
    'Upper_CI_95': confidence_intervals[:, 1],
    'Prediction_Error': actual_prices - predictions,
    'Error_Percentage': ((actual_prices - predictions) / actual_prices * 100)
})

# Prepare metrics
metrics = {
    'MAE': np.mean(np.abs(actual_prices - predictions)),
    'RMSE': np.sqrt(np.mean((actual_prices - predictions) ** 2)),
    'MAPE': np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100,
    'R2': 1 - (np.sum((actual_prices - predictions) ** 2) / 
               np.sum((actual_prices - np.mean(actual_prices)) ** 2)),
    'Directional_Accuracy': np.mean(
        np.sign(np.diff(actual_prices)) == np.sign(np.diff(predictions))
    ) * 100
}

# Create comprehensive report with all plots
report = viz_manager.create_prediction_report(
    predictions=predictions_df,
    metrics=metrics,
    plots=[fig1, fig2, fig3, fig4],
    report_name='demo_viz_comprehensive_report',
    save_html=True,
    save_json=True
)

print(f"  [OK] Comprehensive report created!")
print(f"    - HTML Report: {report['html_path']}")
print(f"    - JSON Report: {report['json_path']}")
print(f"    - Predictions CSV: {report['predictions_csv_path']}")
print(f"    - Plot Files: {len(report['plot_paths'])} images saved")
print()

# ============================================================================
# STEP 8: Display Summary
# ============================================================================
print("=" * 80)
print("VISUALIZATION DEMO SUMMARY")
print("=" * 80)
print()
print("All visualization functions executed successfully!")
print()
print("Generated Visualizations:")
print(f"  1. Time Series Plot with Predictions & Confidence Intervals")
print(f"  2. Feature Importance Plot (Top 15 Features)")
print(f"  3. Economic Indicators Overlay Plot (Multi-Panel)")
print(f"  4. Training History Plot (Loss Curves with Best Epoch)")
print()
print("Generated Reports:")
print(f"  - Comprehensive HTML Report")
print(f"  - JSON Metadata Report")
print(f"  - Predictions CSV Export")
print(f"  - {len(report['plot_paths'])} Plot Images")
print()
print("Performance Metrics:")
print(f"  - MAE:  {metrics['MAE']:.4f} USD")
print(f"  - RMSE: {metrics['RMSE']:.4f} USD")
print(f"  - MAPE: {metrics['MAPE']:.2f}%")
print(f"  - R2:   {metrics['R2']:.4f}")
print(f"  - Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
print()
print(f"All reports saved to: {Config.REPORTS_DIR}")
print()
print("=" * 80)
print("DEMO COMPLETED SUCCESSFULLY!")
print("=" * 80)
print()
print("Next steps:")
print("  1. Open the HTML report in your browser to view the comprehensive report")
print("  2. Review the individual plot images in the reports directory")
print("  3. Check the CSV file for detailed prediction data")
print("  4. Run the test suite: pytest tests/test_visualization_manager.py -v")
print()
