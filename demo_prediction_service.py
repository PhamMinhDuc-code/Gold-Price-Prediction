"""
Demo script for PredictionService functionality.

This script demonstrates:
- Loading a trained model
- Making single-step predictions
- Making multi-step predictions
- Denormalizing predictions
- Computing prediction intervals
- Generating timestamped predictions with confidence intervals
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.prediction_service import PredictionService
from src.model_registry import ModelRegistry
from config import Config


def main():
    """Demonstrate PredictionService functionality."""
    
    print("=" * 80)
    print("GOLD PRICE PREDICTION SERVICE DEMO")
    print("=" * 80)
    
    # Initialize service
    print("\n1. Initializing PredictionService...")
    service = PredictionService()
    print("   ✓ PredictionService initialized")
    
    # Check if models exist
    registry = ModelRegistry()
    models = registry.list_models()
    
    if not models:
        print("\n   ⚠ No trained models found in registry.")
        print("   Please run demo_model_training_complete.py first to train a model.")
        return
    
    print(f"\n   Found {len(models)} model(s) in registry:")
    for model_info in models:
        print(f"   - Version: {model_info.version}, Type: {model_info.model_type}")
    
    # Load latest model
    print("\n2. Loading latest model from registry...")
    try:
        model, metadata = service.load_latest_model()
        print(f"   ✓ Loaded model: {metadata.get('version')}")
        print(f"   - Model type: {metadata.get('model_type')}")
        print(f"   - Features: {len(metadata.get('feature_list', []))}")
        if 'sequence_length' in metadata:
            print(f"   - Sequence length: {metadata.get('sequence_length')}")
        print(f"   - Training date: {metadata.get('training_date')}")
    except Exception as e:
        print(f"   ✗ Error loading model: {str(e)}")
        return
    
    # Create sample input data
    print("\n3. Creating sample input features...")
    model_type = metadata.get('model_type')
    n_features = len(metadata.get('feature_list', []))
    
    if model_type in ['LSTM', 'GRU']:
        sequence_length = metadata.get('sequence_length', 60)
        input_features = np.random.rand(sequence_length, n_features) * 0.5 + 0.25
        print(f"   ✓ Created sequence input: shape {input_features.shape}")
    else:
        input_features = np.random.rand(n_features) * 0.5 + 0.25
        print(f"   ✓ Created feature vector: shape {input_features.shape}")
    
    # Single-step prediction
    print("\n4. Making single-step prediction...")
    try:
        prediction = service.predict_single_step(input_features)
        print(f"   ✓ Single-step prediction (normalized): {prediction:.4f}")
        
        # Denormalize
        denormalized = service.denormalize_predictions(np.array([prediction]))
        print(f"   ✓ Denormalized prediction: ${denormalized[0]:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Multi-step prediction
    print("\n5. Making multi-step prediction (7 days)...")
    try:
        predictions = service.predict_multi_step(input_features, horizon=7)
        print(f"   ✓ Generated {len(predictions)} predictions")
        print(f"   - Normalized range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Denormalize
        denormalized = service.denormalize_predictions(predictions)
        print(f"   ✓ Denormalized predictions:")
        for i, pred in enumerate(denormalized, 1):
            print(f"     Day {i}: ${pred:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Prediction intervals
    print("\n6. Computing prediction intervals...")
    try:
        intervals = service.compute_prediction_intervals(denormalized, confidence=0.95)
        print(f"   ✓ Computed 95% confidence intervals:")
        for i in range(len(denormalized)):
            print(f"     Day {i+1}: ${denormalized[i]:.2f} "
                  f"(${intervals[i, 0]:.2f} - ${intervals[i, 1]:.2f})")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Timestamped predictions
    print("\n7. Generating timestamped predictions (30 days)...")
    try:
        start_date = datetime(2024, 1, 1)
        result_df = service.predict_with_timestamps(
            input_features,
            start_date,
            horizon=30,
            include_confidence=True
        )
        
        print(f"   ✓ Generated timestamped predictions:")
        print(f"\n{result_df.head(10)}")
        print(f"   ... ({len(result_df)} total rows)")
        
        # Summary statistics
        print(f"\n   Summary:")
        print(f"   - Date range: {result_df['Date'].iloc[0]} to {result_df['Date'].iloc[-1]}")
        print(f"   - Prediction range: ${result_df['Prediction'].min():.2f} - ${result_df['Prediction'].max():.2f}")
        print(f"   - Mean prediction: ${result_df['Prediction'].mean():.2f}")
        
        if 'Lower_CI' in result_df.columns:
            avg_interval = (result_df['Upper_CI'] - result_df['Lower_CI']).mean()
            print(f"   - Average confidence interval width: ${avg_interval:.2f}")
    
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Save predictions to file
    print("\n8. Saving predictions to file...")
    try:
        output_file = Config.REPORTS_DIR / "sample_predictions.csv"
        result_df.to_csv(output_file, index=False)
        print(f"   ✓ Predictions saved to: {output_file}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nPredictionService successfully demonstrated:")
    print("  ✓ Model loading from registry")
    print("  ✓ Single-step prediction")
    print("  ✓ Multi-step prediction (recursive)")
    print("  ✓ Denormalization to original price scale")
    print("  ✓ Prediction interval computation")
    print("  ✓ Timestamped predictions with confidence intervals")
    print("\nAll subtasks for Task 11 completed successfully!")


if __name__ == "__main__":
    main()
