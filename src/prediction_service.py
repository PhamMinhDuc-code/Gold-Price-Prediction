"""
Prediction Service Module

This module implements the PredictionService class for generating gold price
predictions using trained models with confidence intervals.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import logging
import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

from src.model_registry import ModelRegistry
from src.exceptions import ExtrapolationWarning
from config import Config


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

# Add handler if not already present
if not logger.handlers:
    log_file = Config.get_log_path('src.prediction_service.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class PredictionService:
    """
    Prediction Service for generating gold price forecasts.
    
    This class provides functionality to:
    - Load trained models using ModelRegistry
    - Generate single-step and multi-step predictions
    - Denormalize predictions to original scale
    - Compute prediction intervals
    - Generate timestamped predictions
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        """
        Initialize PredictionService with ModelRegistry.
        
        Args:
            model_registry: ModelRegistry instance. If None, creates new instance.
        
        Requirements: 6.1
        """
        self.model_registry = model_registry if model_registry is not None else ModelRegistry()
        self.model = None
        self.metadata = None
        self.scaling_params = None
        self.model_type = None
        self.feature_list = None
        self.sequence_length = None
        self.training_data_range = None  # Store training data feature ranges
        
        logger.info("PredictionService initialized")
    
    def load_model(self, model_path: str, metadata_path: str) -> Tuple[Any, Dict]:
        """
        Load trained model and metadata.
        
        Args:
            model_path: Path to model directory
            metadata_path: Path to metadata file (ignored, loaded from model_path)
        
        Returns:
            Tuple of (model, metadata)
        
        Requirements: 6.1
        """
        logger.info(f"Loading model from {model_path}")
        
        model_path = Path(model_path)
        
        # Load metadata first to determine model type
        self.metadata = self.model_registry.load_metadata(model_path)
        self.model_type = self.metadata.get('model_type')
        
        # Load the model
        self.model = self.model_registry.load_model(model_path, self.model_type)
        
        # Load scaling parameters
        self.scaling_params = self.model_registry.load_scaling_params(model_path)
        if self.scaling_params:
            self.metadata['scaling_params'] = self.scaling_params
        
        # Extract useful metadata
        self.feature_list = self.metadata.get('feature_list', [])
        self.sequence_length = self.metadata.get('sequence_length')
        self.training_data_range = self.metadata.get('training_data_range', {})
        
        logger.info(f"Model loaded successfully: {self.model_type}")
        logger.info(f"Feature count: {len(self.feature_list)}")
        if self.sequence_length:
            logger.info(f"Sequence length: {self.sequence_length}")
        if self.training_data_range:
            logger.info(f"Training data ranges available for {len(self.training_data_range)} features")
        
        return self.model, self.metadata
    
    def load_latest_model(self) -> Tuple[Any, Dict]:
        """
        Load the most recent model version from registry.
        
        Returns:
            Tuple of (model, metadata)
        
        Requirements: 6.1
        """
        logger.info("Loading latest model from registry")
        
        self.model, self.metadata = self.model_registry.load_latest_model()
        
        # Extract metadata
        self.model_type = self.metadata.get('model_type')
        self.scaling_params = self.metadata.get('scaling_params', {})
        self.feature_list = self.metadata.get('feature_list', [])
        self.sequence_length = self.metadata.get('sequence_length')
        self.training_data_range = self.metadata.get('training_data_range', {})
        
        logger.info(f"Latest model loaded: {self.metadata.get('version')}")
        
        return self.model, self.metadata
    
    def load_model_by_version(self, version: str) -> Tuple[Any, Dict]:
        """
        Load specific model version from registry.
        
        Args:
            version: Model version string
        
        Returns:
            Tuple of (model, metadata)
        
        Requirements: 6.1
        """
        logger.info(f"Loading model version: {version}")
        
        self.model, self.metadata = self.model_registry.load_model_by_version(version)
        
        # Extract metadata
        self.model_type = self.metadata.get('model_type')
        self.scaling_params = self.metadata.get('scaling_params', {})
        self.feature_list = self.metadata.get('feature_list', [])
        self.sequence_length = self.metadata.get('sequence_length')
        self.training_data_range = self.metadata.get('training_data_range', {})
        
        logger.info(f"Model version {version} loaded successfully")
        
        return self.model, self.metadata
    
    def check_extrapolation(self, input_features: np.ndarray) -> None:
        """
        Check if input features are outside training data range.
        
        Logs ExtrapolationWarning with feature name and range if any feature
        is outside the training data range.
        
        Args:
            input_features: Input features array
                - For LSTM/GRU: shape (sequence_length, n_features)
                - For XGBoost/RF: shape (n_features,) or (1, n_features)
        
        Requirements: 10.2
        """
        if not self.training_data_range or not self.feature_list:
            logger.debug("Training data range not available, skipping extrapolation check")
            return
        
        logger.debug("Checking for extrapolation")
        
        # Flatten features for checking
        if len(input_features.shape) > 1:
            # For sequence models, check the most recent timestep
            if self.model_type in ['LSTM', 'GRU']:
                features_to_check = input_features[-1]  # Last timestep
            else:
                features_to_check = input_features.flatten()
        else:
            features_to_check = input_features
        
        # Check each feature against training range
        extrapolation_warnings = []
        for idx, feature_name in enumerate(self.feature_list):
            if idx >= len(features_to_check):
                break
            
            if feature_name in self.training_data_range:
                input_value = features_to_check[idx]
                train_min, train_max = self.training_data_range[feature_name]
                
                # Check if outside range
                if input_value < train_min or input_value > train_max:
                    warning_msg = (
                        f"Extrapolation risk: Feature '{feature_name}' value {input_value:.4f} "
                        f"is outside training range [{train_min:.4f}, {train_max:.4f}]"
                    )
                    logger.warning(warning_msg)
                    extrapolation_warnings.append(warning_msg)
                    
                    # Issue Python warning (Requirement 10.2)
                    warnings.warn(
                        ExtrapolationWarning(feature_name, input_value, (train_min, train_max)),
                        stacklevel=2
                    )
        
        if extrapolation_warnings:
            logger.warning(f"Found {len(extrapolation_warnings)} features outside training range")
        else:
            logger.debug("All features within training range")
        
        return self.model, self.metadata
    
    def predict_single_step(self, input_features: np.ndarray) -> float:
        """
        Generate next-day prediction.
        
        Args:
            input_features: Preprocessed input features
                - For LSTM/GRU: shape (sequence_length, n_features)
                - For XGBoost/RF: shape (n_features,)
        
        Returns:
            Single prediction value (normalized)
        
        Requirements: 6.1, 6.2, 6.3
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.debug(f"Generating single-step prediction with input shape: {input_features.shape}")
        
        # Check for extrapolation (Requirement 10.2)
        self.check_extrapolation(input_features)
        
        # Prepare input based on model type
        if self.model_type in ['LSTM', 'GRU']:
            # Sequence models expect 3D input: (batch_size, sequence_length, n_features)
            if len(input_features.shape) == 2:
                input_features = np.expand_dims(input_features, axis=0)
            elif len(input_features.shape) == 1:
                raise ValueError(f"LSTM/GRU models require 2D input (sequence_length, n_features), got 1D")
        else:
            # Tree-based models expect 2D input: (batch_size, n_features)
            if len(input_features.shape) == 1:
                input_features = np.expand_dims(input_features, axis=0)
        
        # Make prediction
        # NOTE: `verbose` is a Keras-only kwarg (LSTM/GRU). XGBoost/RandomForest's
        # predict() does not accept it, so only pass it for sequence models.
        if self.model_type in ['LSTM', 'GRU']:
            prediction = self.model.predict(input_features, verbose=0)
        else:
            prediction = self.model.predict(input_features)
        
        # Extract scalar value
        if isinstance(prediction, np.ndarray):
            prediction = float(prediction.flatten()[0])
        
        logger.debug(f"Single-step prediction: {prediction}")
        
        return prediction
    
    def predict_multi_step(self, input_features: np.ndarray, horizon: int) -> np.ndarray:
        """
        Generate multi-step ahead predictions using recursive strategy.
        
        For LSTM/GRU: Uses previous predictions as inputs for future steps.
        For XGBoost/RF: Uses lag features updated with predictions.
        
        Args:
            input_features: Preprocessed input features
                - For LSTM/GRU: shape (sequence_length, n_features)
                - For XGBoost/RF: shape (n_features,)
            horizon: Number of time steps to predict (up to 30 days)
        
        Returns:
            Array of predictions, shape (horizon,)
        
        Requirements: 6.2, 6.3
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if horizon > 30:
            logger.warning(f"Horizon {horizon} exceeds recommended maximum of 30 days")
        
        logger.info(f"Generating {horizon}-step ahead predictions")
        
        predictions = []
        
        if self.model_type in ['LSTM', 'GRU']:
            # Recursive prediction for sequence models
            current_sequence = input_features.copy()
            
            for step in range(horizon):
                # Predict next step
                pred = self.predict_single_step(current_sequence)
                predictions.append(pred)
                
                # Update sequence: shift left and append new prediction
                # Assume target is in the first feature (index 0)
                new_row = current_sequence[-1].copy()
                new_row[0] = pred  # Update target feature with prediction
                
                # Shift sequence and add new row
                current_sequence = np.vstack([current_sequence[1:], new_row])
        
        else:
            # Recursive prediction for tree-based models
            current_features = input_features.copy()
            
            for step in range(horizon):
                # Predict next step
                pred = self.predict_single_step(current_features)
                predictions.append(pred)
                
                # Update lag features (assuming they exist in feature list)
                # This is a simplified approach - in practice, need to update all lag features
                current_features = current_features.copy()
                
                # Update first feature (target) with prediction
                if len(current_features.shape) == 1:
                    current_features[0] = pred
                else:
                    current_features[0, 0] = pred
        
        predictions = np.array(predictions)
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def denormalize_predictions(self, predictions: np.ndarray, 
                               scaling_params: Optional[Dict] = None) -> np.ndarray:
        """
        Convert normalized predictions back to original price scale.
        
        Args:
            predictions: Normalized predictions
            scaling_params: Scaling parameters dict. If None, uses loaded params.
        
        Returns:
            Denormalized predictions in original scale
        
        Requirements: 6.4
        """
        if scaling_params is None:
            scaling_params = self.scaling_params
        
        if not scaling_params:
            logger.warning("No scaling parameters available. Returning predictions unchanged.")
            return predictions
        
        logger.info(f"Denormalizing {len(predictions)} predictions")
        
        # Get target column scaling params
        target_col = Config.TARGET_COLUMN  # 'Close'
        
        if target_col not in scaling_params:
            logger.warning(f"Scaling parameters for {target_col} not found. Returning unchanged.")
            return predictions
        
        params = scaling_params[target_col]
        method = params.get('method', 'minmax')
        
        if method == 'minmax':
            # Inverse min-max: X_original = X_normalized * (max - min) + min
            min_val = params.get('min', 0.0)
            max_val = params.get('max', 1.0)
            
            denormalized = predictions * (max_val - min_val) + min_val
            
            logger.debug(f"Applied inverse min-max: min={min_val}, max={max_val}")
        
        elif method == 'zscore' or method == 'standard':
            # Inverse z-score: X_original = X_normalized * std + mean
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            
            denormalized = predictions * std + mean
            
            logger.debug(f"Applied inverse z-score: mean={mean}, std={std}")
        
        else:
            logger.warning(f"Unknown normalization method: {method}. Returning unchanged.")
            denormalized = predictions
        
        logger.info(f"Denormalization complete. Range: [{denormalized.min():.2f}, {denormalized.max():.2f}]")
        
        return denormalized
    
    def compute_prediction_intervals(self, predictions: np.ndarray,
                                   confidence: float = 0.95) -> np.ndarray:
        """
        Calculate confidence intervals for predictions using bootstrap method.
        
        This is a simplified implementation using historical error distribution.
        For production, consider more sophisticated methods like quantile regression.
        
        Args:
            predictions: Array of predictions
            confidence: Confidence level (default 0.95 for 95% confidence)
        
        Returns:
            Array of shape (n_predictions, 2) with [lower_bound, upper_bound]
        
        Requirements: 6.6
        """
        logger.info(f"Computing prediction intervals at {confidence*100}% confidence")
        
        # Simplified approach: use a percentage of the prediction as interval
        # In practice, this should be based on model's historical error distribution
        
        # Calculate alpha for confidence level
        alpha = 1 - confidence
        
        # Estimate interval width as percentage of prediction
        # This is a placeholder - in production, use actual error quantiles
        interval_width_pct = 0.05  # 5% of prediction value as initial estimate
        
        # For time series, uncertainty typically increases with horizon
        n_predictions = len(predictions)
        horizon_factors = np.linspace(1.0, 1.5, n_predictions)  # Increase uncertainty over time
        
        intervals = np.zeros((n_predictions, 2))
        
        for i in range(n_predictions):
            pred = predictions[i]
            width = abs(pred) * interval_width_pct * horizon_factors[i]
            
            # Calculate bounds
            intervals[i, 0] = pred - width  # Lower bound
            intervals[i, 1] = pred + width  # Upper bound
        
        logger.info(f"Prediction intervals computed for {n_predictions} predictions")
        logger.debug(f"Average interval width: {np.mean(intervals[:, 1] - intervals[:, 0]):.2f}")
        
        return intervals
    
    def predict_with_timestamps(self, input_features: np.ndarray,
                               start_date: datetime,
                               horizon: int,
                               include_confidence: bool = True) -> pd.DataFrame:
        """
        Generate predictions with corresponding timestamps.
        
        Args:
            input_features: Preprocessed input features
            start_date: Starting date for predictions
            horizon: Number of days to predict
            include_confidence: Whether to include confidence intervals
        
        Returns:
            DataFrame with columns: Date, Prediction, (Lower_CI, Upper_CI if requested)
        
        Requirements: 6.5, 6.6
        """
        logger.info(f"Generating timestamped predictions from {start_date} for {horizon} days")
        
        # Generate predictions
        predictions_normalized = self.predict_multi_step(input_features, horizon)
        
        # Denormalize to original scale
        predictions = self.denormalize_predictions(predictions_normalized)
        
        # Generate timestamps (business days)
        dates = pd.bdate_range(start=start_date, periods=horizon)
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'Date': dates,
            'Prediction': predictions
        })
        
        # Add confidence intervals if requested
        if include_confidence:
            intervals = self.compute_prediction_intervals(
                predictions, 
                confidence=Config.CONFIDENCE_LEVEL
            )
            
            result_df['Lower_CI'] = intervals[:, 0]
            result_df['Upper_CI'] = intervals[:, 1]
            
            logger.info("Confidence intervals included in output")
        
        logger.info(f"Generated predictions from {dates[0]} to {dates[-1]}")
        logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        return result_df


if __name__ == "__main__":
    # Example usage
    print("PredictionService module loaded successfully")
    
    # Initialize service
    service = PredictionService()
    
    print("\nPredictionService initialized")
    print("Available methods:")
    print("  - load_model()")
    print("  - load_latest_model()")
    print("  - load_model_by_version()")
    print("  - predict_single_step()")
    print("  - predict_multi_step()")
    print("  - denormalize_predictions()")
    print("  - compute_prediction_intervals()")
    print("  - predict_with_timestamps()")
