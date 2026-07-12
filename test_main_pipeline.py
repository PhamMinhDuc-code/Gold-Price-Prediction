"""
Test script for main pipeline orchestration.

This script tests the main.py pipeline in all three modes:
- Training mode
- Prediction mode
- Evaluation mode
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from main import GoldPricePipeline
from config import Config
from src.logger import get_logger

logger = get_logger(__name__)


def create_test_data(filepath: str, n_samples: int = 500):
    """Create synthetic test data for demonstration."""
    logger.info(f"Creating test data: {n_samples} samples")
    
    # Generate date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='B')
    
    # Generate synthetic gold prices with trend and seasonality
    np.random.seed(42)
    trend = np.linspace(1500, 1800, n_samples)
    seasonal = 50 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    noise = np.random.randn(n_samples) * 20
    
    close_prices = trend + seasonal + noise
    
    # Generate OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(n_samples) * 5,
        'High': close_prices + np.abs(np.random.randn(n_samples) * 10),
        'Low': close_prices - np.abs(np.random.randn(n_samples) * 10),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, n_samples)
    })
    
    # Ensure OHLC constraints
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) + 5
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) - 5
    
    # Format dates
    df['Date'] = df['Date'].dt.strftime('%Y.%m.%d %H:%M')
    
    # Save to CSV
    df.to_csv(filepath, sep=';', index=False)
    logger.info(f"Test data saved to {filepath}")
    
    return df


def test_training_pipeline():
    """Test training pipeline."""
    print("=" * 80)
    print("TEST 1: TRAINING PIPELINE")
    print("=" * 80)
    
    # Create test data
    data_path = Config.DATA_DIR / "test_gold_data.csv"
    create_test_data(str(data_path), n_samples=500)
    
    # Initialize pipeline
    pipeline = GoldPricePipeline()
    
    # Run training pipeline (use RandomForest for faster testing - no TensorFlow required)
    try:
        model_path = pipeline.run_training_pipeline(
            data_path=str(data_path),
            model_type='RandomForest',
            version='test_v1.0',
            indicators_config=None,  # Skip indicators for faster testing
            hyperparams={'n_estimators': 10, 'max_depth': 5},  # Fast hyperparams
            tune_hyperparams=False
        )
        
        print(f"\n✓ Training pipeline completed successfully!")
        print(f"  Model saved to: {model_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Training pipeline failed: {str(e)}")
        logger.error(f"Training test failed", exc_info=True)
        return False


def test_prediction_pipeline():
    """Test prediction pipeline."""
    print("\n" + "=" * 80)
    print("TEST 2: PREDICTION PIPELINE")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = GoldPricePipeline()
    
    # Run prediction pipeline
    try:
        predictions_df = pipeline.run_prediction_pipeline(
            model_version='test_v1.0',
            input_data=None,
            horizon=10,
            output_path=str(Config.REPORTS_DIR / "test_predictions.csv"),
            include_confidence=True
        )
        
        print(f"\n✓ Prediction pipeline completed successfully!")
        print(f"  Generated {len(predictions_df)} predictions")
        print(f"\n  First few predictions:")
        print(predictions_df.head())
        return True
        
    except Exception as e:
        print(f"\n✗ Prediction pipeline failed: {str(e)}")
        logger.error(f"Prediction test failed", exc_info=True)
        return False


def test_evaluation_pipeline():
    """Test evaluation pipeline."""
    print("\n" + "=" * 80)
    print("TEST 3: EVALUATION PIPELINE")
    print("=" * 80)
    
    # Create test data
    test_data_path = Config.DATA_DIR / "test_eval_data.csv"
    create_test_data(str(test_data_path), n_samples=200)
    
    # Initialize pipeline
    pipeline = GoldPricePipeline()
    
    # Run evaluation pipeline
    try:
        result = pipeline.run_evaluation_pipeline(
            model_version='test_v1.0',
            test_data_path=str(test_data_path),
            compare_version=None
        )
        
        print(f"\n✓ Evaluation pipeline completed successfully!")
        print(f"\n  Performance Metrics:")
        for metric, value in result['metrics'].items():
            print(f"    {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Evaluation pipeline failed: {str(e)}")
        logger.error(f"Evaluation test failed", exc_info=True)
        return False


def main():
    """Run all pipeline tests."""
    print("\n" + "=" * 80)
    print("MAIN PIPELINE ORCHESTRATION TESTS")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    results = {
        'Training': False,
        'Prediction': False,
        'Evaluation': False
    }
    
    # Test 1: Training
    results['Training'] = test_training_pipeline()
    
    # Test 2: Prediction (only if training succeeded)
    if results['Training']:
        results['Prediction'] = test_prediction_pipeline()
    else:
        print("\n⚠ Skipping prediction test (training failed)")
    
    # Test 3: Evaluation (only if training succeeded)
    if results['Training']:
        results['Evaluation'] = test_evaluation_pipeline()
    else:
        print("\n⚠ Skipping evaluation test (training failed)")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)
    
    print(f"\nCompleted at: {datetime.now()}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
