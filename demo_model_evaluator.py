"""
Demo script for ModelEvaluator module.

This script demonstrates:
1. Creating sample predictions and actual values
2. Calculating all evaluation metrics
3. Generating residual plots
4. Generating prediction vs actual plots
5. Creating comprehensive performance reports

Requirements: 7.1-7.7
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.model_evaluator import ModelEvaluator
from config import Config


def create_sample_data(n_samples=100, noise_level=5.0, trend=True):
    """
    Create sample gold price predictions for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Standard deviation of prediction noise
        trend: Whether to include an upward trend
    
    Returns:
        Tuple of (y_true, y_pred, dates)
    """
    print(f"\nGenerating {n_samples} sample data points...")
    
    # Create dates (business days)
    start_date = datetime(2024, 1, 1)
    dates = pd.bdate_range(start=start_date, periods=n_samples)
    
    # Generate realistic gold prices with trend and seasonality
    np.random.seed(42)
    
    # Base price
    base_price = 2000.0
    
    # Add trend
    if trend:
        trend_component = np.linspace(0, 100, n_samples)
    else:
        trend_component = 0
    
    # Add seasonality (quarterly cycle)
    time_index = np.arange(n_samples)
    seasonal_component = 30 * np.sin(2 * np.pi * time_index / 90)
    
    # Add random walk
    random_walk = np.cumsum(np.random.randn(n_samples) * 2)
    
    # Generate actual values
    y_true = base_price + trend_component + seasonal_component + random_walk
    
    # Generate predictions with some error
    prediction_noise = np.random.randn(n_samples) * noise_level
    y_pred = y_true + prediction_noise
    
    print(f"Actual price range: ${y_true.min():.2f} - ${y_true.max():.2f}")
    print(f"Predicted price range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    
    return y_true, y_pred, dates.to_list()


def demo_metric_calculations():
    """Demonstrate metric calculations."""
    print("\n" + "=" * 70)
    print("DEMO 1: Metric Calculations")
    print("=" * 70)
    
    # Create sample data
    y_true, y_pred, dates = create_sample_data(n_samples=200, noise_level=10.0)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(y_true, y_pred, dates)
    
    # Calculate individual metrics
    print("\n1. Calculating individual metrics:")
    print("-" * 70)
    
    mae = evaluator.calculate_mae()
    print(f"   MAE (Mean Absolute Error): ${mae:.2f}")
    print(f"   → Average prediction error magnitude")
    
    rmse = evaluator.calculate_rmse()
    print(f"\n   RMSE (Root Mean Squared Error): ${rmse:.2f}")
    print(f"   → Penalizes large errors more than MAE")
    
    mape = evaluator.calculate_mape()
    print(f"\n   MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"   → Scale-independent error metric")
    
    r2 = evaluator.calculate_r2()
    print(f"\n   R² (Coefficient of Determination): {r2:.4f}")
    print(f"   → Proportion of variance explained (1.0 = perfect)")
    
    accuracy = evaluator.calculate_directional_accuracy()
    print(f"\n   Directional Accuracy: {accuracy:.2f}%")
    print(f"   → Percentage of correct up/down predictions")
    
    # Calculate all at once
    print("\n2. Calculating all metrics at once:")
    print("-" * 70)
    
    all_metrics = evaluator.calculate_all_metrics()
    for metric_name, value in all_metrics.items():
        print(f"   {metric_name}: {value:.4f}")


def demo_plot_generation():
    """Demonstrate plot generation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Plot Generation")
    print("=" * 70)
    
    # Create sample data
    y_true, y_pred, dates = create_sample_data(n_samples=150, noise_level=8.0)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(y_true, y_pred, dates)
    
    # Generate residual plot
    print("\n1. Generating residual plot...")
    residual_path = Config.REPORTS_DIR / "demo_residuals.png"
    evaluator.plot_residuals(save_path=residual_path, show=False)
    print(f"   ✓ Residual plot saved to: {residual_path}")
    
    # Generate predictions vs actual plot
    print("\n2. Generating predictions vs actual plot...")
    pred_plot_path = Config.REPORTS_DIR / "demo_predictions_vs_actual.png"
    evaluator.plot_predictions_vs_actual(save_path=pred_plot_path, show=False)
    print(f"   ✓ Predictions plot saved to: {pred_plot_path}")
    
    print("\n   Note: Open the PNG files to view the plots")


def demo_performance_report():
    """Demonstrate comprehensive performance report generation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Comprehensive Performance Report")
    print("=" * 70)
    
    # Create sample data
    y_true, y_pred, dates = create_sample_data(n_samples=250, noise_level=12.0)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(y_true, y_pred, dates)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive performance report...")
    report = evaluator.generate_performance_report(
        report_name="demo_evaluation_report",
        save_json=True,
        save_plots=True
    )
    
    print("\n" + "-" * 70)
    print("Report Generated Successfully!")
    print("-" * 70)
    print(f"\nReport Name: {report['report_name']}")
    print(f"Generated At: {report['generated_at']}")
    print(f"Sample Count: {report['sample_count']}")
    
    print("\nPerformance Summary:")
    print(f"  • MAE:  ${report['summary']['mae']:.2f}")
    print(f"  • RMSE: ${report['summary']['rmse']:.2f}")
    print(f"  • MAPE: {report['summary']['mape']:.2f}%")
    print(f"  • R²:   {report['summary']['r2']:.4f}")
    print(f"  • Directional Accuracy: {report['summary']['directional_accuracy']:.2f}%")
    
    if 'date_range' in report and report['date_range']:
        print(f"\nDate Range:")
        print(f"  • Start: {report['date_range']['start']}")
        print(f"  • End:   {report['date_range']['end']}")
    
    print("\nGenerated Files:")
    if 'json_path' in report:
        print(f"  • JSON Report: {report['json_path']}")
    if 'residual_plot_path' in report:
        print(f"  • Residual Plot: {report['residual_plot_path']}")
    if 'predictions_plot_path' in report:
        print(f"  • Predictions Plot: {report['predictions_plot_path']}")


def demo_comparison_scenarios():
    """Demonstrate different prediction quality scenarios."""
    print("\n" + "=" * 70)
    print("DEMO 4: Comparing Different Model Quality Scenarios")
    print("=" * 70)
    
    scenarios = [
        ("Excellent Model", 2.0),
        ("Good Model", 5.0),
        ("Average Model", 10.0),
        ("Poor Model", 20.0)
    ]
    
    print("\nComparing models with different prediction quality:\n")
    
    results = []
    
    for scenario_name, noise_level in scenarios:
        # Generate data with different noise levels
        y_true, y_pred, _ = create_sample_data(n_samples=100, noise_level=noise_level, trend=False)
        
        # Evaluate
        evaluator = ModelEvaluator(y_true, y_pred)
        metrics = evaluator.calculate_all_metrics()
        
        results.append({
            'Scenario': scenario_name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'R²': metrics['R2'],
            'Dir_Acc': metrics['Directional_Accuracy']
        })
    
    # Display comparison table
    print(f"{'Scenario':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10} {'Dir_Acc':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['Scenario']:<20} "
              f"${result['MAE']:<9.2f} "
              f"${result['RMSE']:<9.2f} "
              f"{result['MAPE']:<9.2f}% "
              f"{result['R²']:<9.4f} "
              f"{result['Dir_Acc']:<9.1f}%")
    
    print("\nObservations:")
    print("  • Lower MAE/RMSE/MAPE values indicate better predictions")
    print("  • Higher R² (closer to 1.0) indicates better model fit")
    print("  • Higher Directional Accuracy indicates better trend prediction")


def demo_edge_cases():
    """Demonstrate edge cases and error handling."""
    print("\n" + "=" * 70)
    print("DEMO 5: Edge Cases and Error Handling")
    print("=" * 70)
    
    # Perfect predictions
    print("\n1. Perfect predictions (zero error):")
    y_true = np.array([100.0, 110.0, 105.0, 115.0, 120.0])
    y_pred = y_true.copy()
    
    evaluator = ModelEvaluator(y_true, y_pred)
    metrics = evaluator.calculate_all_metrics()
    
    print(f"   MAE: {metrics['MAE']:.6f} (should be ~0)")
    print(f"   RMSE: {metrics['RMSE']:.6f} (should be ~0)")
    print(f"   R²: {metrics['R2']:.6f} (should be 1.0)")
    
    # Constant predictions
    print("\n2. Constant predictions (predicting mean):")
    y_true = np.array([100.0, 110.0, 105.0, 115.0, 120.0])
    y_pred = np.array([110.0, 110.0, 110.0, 110.0, 110.0])
    
    evaluator = ModelEvaluator(y_true, y_pred)
    metrics = evaluator.calculate_all_metrics()
    
    print(f"   MAE: {metrics['MAE']:.2f}")
    print(f"   R²: {metrics['R2']:.6f} (should be ~0)")
    print(f"   Directional Accuracy: {metrics['Directional_Accuracy']:.1f}% (should be 0)")
    
    # Opposite directions
    print("\n3. Predictions with opposite trend:")
    y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0])  # Increasing
    y_pred = np.array([140.0, 130.0, 120.0, 110.0, 100.0])  # Decreasing
    
    evaluator = ModelEvaluator(y_true, y_pred)
    metrics = evaluator.calculate_all_metrics()
    
    print(f"   MAE: {metrics['MAE']:.2f}")
    print(f"   R²: {metrics['R2']:.4f} (should be negative)")
    print(f"   Directional Accuracy: {metrics['Directional_Accuracy']:.1f}% (should be 0)")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATOR MODULE - DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases the ModelEvaluator functionality:")
    print("  • Calculating regression metrics (MAE, RMSE, MAPE, R²)")
    print("  • Calculating directional accuracy")
    print("  • Generating residual plots")
    print("  • Generating prediction vs actual plots")
    print("  • Creating comprehensive performance reports")
    
    # Run demos
    demo_metric_calculations()
    demo_plot_generation()
    demo_performance_report()
    demo_comparison_scenarios()
    demo_edge_cases()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAll demonstrations completed successfully!")
    print(f"\nGenerated files are located in: {Config.REPORTS_DIR}")
    print("\nYou can now:")
    print("  1. Review the generated plots in the reports directory")
    print("  2. Examine the JSON performance report")
    print("  3. Use ModelEvaluator in your own model evaluation workflows")


if __name__ == "__main__":
    main()
