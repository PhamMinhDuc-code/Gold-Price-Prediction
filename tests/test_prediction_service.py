"""
Unit tests for PredictionService class.

Tests cover:
- Single-step prediction
- Multi-step prediction
- Denormalization
- Prediction intervals
- Timestamped predictions

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction_service import PredictionService
from src.model_registry import ModelRegistry
from config import Config


class TestPredictionServiceInitialization:
    """Test PredictionService initialization."""
    
    def test_init_default_registry(self):
        """Test initialization with default ModelRegistry."""
        service = PredictionService()
        
        assert service.model_registry is not None
        assert isinstance(service.model_registry, ModelRegistry)
        assert service.model is None
        assert service.metadata is None
    
    def test_init_custom_registry(self):
        """Test initialization with custom ModelRegistry."""
        custom_registry = ModelRegistry()
        service = PredictionService(model_registry=custom_registry)
        
        assert service.model_registry is custom_registry


class TestModelLoading:
    """Test model loading functionality."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock ModelRegistry."""
        registry = Mock(spec=ModelRegistry)
        return registry
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return {
            'version': 'v1.0.0',
            'model_type': 'LSTM',
            'feature_list': ['Close', 'Volume', 'Close_lag_1'],
            'sequence_length': 60,
            'scaling_params': {
                'Close': {'method': 'minmax', 'min': 1500.0, 'max': 2000.0}
            }
        }
    
    def test_load_model(self, mock_registry, sample_metadata):
        """Test load_model method."""
        # Setup mocks
        mock_model = Mock()
        mock_registry.load_metadata.return_value = sample_metadata
        mock_registry.load_model.return_value = mock_model
        mock_registry.load_scaling_params.return_value = sample_metadata['scaling_params']
        
        # Test
        service = PredictionService(model_registry=mock_registry)
        model, metadata = service.load_model('models/test', 'metadata.json')
        
        # Verify
        assert model is mock_model
        assert metadata == sample_metadata
        assert service.model_type == 'LSTM'
        assert service.sequence_length == 60
        assert len(service.feature_list) == 3
    
    def test_load_latest_model(self, mock_registry, sample_metadata):
        """Test load_latest_model method."""
        # Setup mocks
        mock_model = Mock()
        mock_registry.load_latest_model.return_value = (mock_model, sample_metadata)
        
        # Test
        service = PredictionService(model_registry=mock_registry)
        model, metadata = service.load_latest_model()
        
        # Verify
        assert model is mock_model
        assert metadata == sample_metadata
        mock_registry.load_latest_model.assert_called_once()
    
    def test_load_model_by_version(self, mock_registry, sample_metadata):
        """Test load_model_by_version method."""
        # Setup mocks
        mock_model = Mock()
        mock_registry.load_model_by_version.return_value = (mock_model, sample_metadata)
        
        # Test
        service = PredictionService(model_registry=mock_registry)
        model, metadata = service.load_model_by_version('v1.0.0')
        
        # Verify
        assert model is mock_model
        assert metadata == sample_metadata
        mock_registry.load_model_by_version.assert_called_once_with('v1.0.0')


class TestSingleStepPrediction:
    """Test single-step prediction functionality."""
    
    def test_predict_single_step_lstm(self):
        """Test single-step prediction for LSTM model."""
        # Create service with mock model
        service = PredictionService()
        service.model = Mock()
        service.model.predict.return_value = np.array([[0.5]])
        service.model_type = 'LSTM'
        service.sequence_length = 60
        
        # Create input (sequence_length, n_features)
        input_features = np.random.rand(60, 5)
        
        # Test
        prediction = service.predict_single_step(input_features)
        
        # Verify
        assert isinstance(prediction, float)
        assert prediction == 0.5
        service.model.predict.assert_called_once()
    
    def test_predict_single_step_xgboost(self):
        """Test single-step prediction for XGBoost model."""
        # Create service with mock model
        service = PredictionService()
        service.model = Mock()
        service.model.predict.return_value = np.array([0.75])
        service.model_type = 'XGBoost'
        
        # Create input (n_features,)
        input_features = np.random.rand(20)
        
        # Test
        prediction = service.predict_single_step(input_features)
        
        # Verify
        assert isinstance(prediction, float)
        assert prediction == 0.75
    
    def test_predict_single_step_no_model(self):
        """Test prediction fails when no model is loaded."""
        service = PredictionService()
        
        with pytest.raises(ValueError, match="No model loaded"):
            service.predict_single_step(np.array([1, 2, 3]))


class TestMultiStepPrediction:
    """Test multi-step prediction functionality."""
    
    def test_predict_multi_step_lstm(self):
        """Test multi-step prediction for LSTM model."""
        # Create service with mock model
        service = PredictionService()
        service.model = Mock()
        # Return different values for each call
        service.model.predict.side_effect = [
            np.array([[0.5]]),
            np.array([[0.52]]),
            np.array([[0.54]])
        ]
        service.model_type = 'LSTM'
        service.sequence_length = 60
        
        # Create input
        input_features = np.random.rand(60, 5)
        
        # Test 3-step prediction
        predictions = service.predict_multi_step(input_features, horizon=3)
        
        # Verify
        assert len(predictions) == 3
        assert predictions[0] == 0.5
        assert predictions[1] == 0.52
        assert predictions[2] == 0.54
        assert service.model.predict.call_count == 3
    
    def test_predict_multi_step_xgboost(self):
        """Test multi-step prediction for XGBoost model."""
        # Create service with mock model
        service = PredictionService()
        service.model = Mock()
        service.model.predict.side_effect = [
            np.array([0.6]),
            np.array([0.65]),
            np.array([0.7]),
            np.array([0.72]),
            np.array([0.75])
        ]
        service.model_type = 'XGBoost'
        
        # Create input
        input_features = np.random.rand(20)
        
        # Test 5-step prediction
        predictions = service.predict_multi_step(input_features, horizon=5)
        
        # Verify
        assert len(predictions) == 5
        assert all(isinstance(p, (float, np.floating)) for p in predictions)
        assert service.model.predict.call_count == 5
    
    def test_predict_multi_step_long_horizon(self):
        """Test multi-step prediction with long horizon (>30 days)."""
        service = PredictionService()
        service.model = Mock()
        service.model.predict.return_value = np.array([[0.5]])
        service.model_type = 'LSTM'
        
        input_features = np.random.rand(60, 5)
        
        # Should warn but still work
        predictions = service.predict_multi_step(input_features, horizon=50)
        
        assert len(predictions) == 50


class TestDenormalization:
    """Test denormalization functionality."""
    
    def test_denormalize_minmax(self):
        """Test denormalization with min-max scaling."""
        service = PredictionService()
        
        # Setup scaling params
        scaling_params = {
            'Close': {
                'method': 'minmax',
                'min': 1500.0,
                'max': 2000.0
            }
        }
        
        # Normalized predictions (0 to 1 scale)
        normalized = np.array([0.0, 0.5, 1.0])
        
        # Test
        denormalized = service.denormalize_predictions(normalized, scaling_params)
        
        # Verify
        assert np.allclose(denormalized, [1500.0, 1750.0, 2000.0])
    
    def test_denormalize_zscore(self):
        """Test denormalization with z-score scaling."""
        service = PredictionService()
        
        # Setup scaling params
        scaling_params = {
            'Close': {
                'method': 'zscore',
                'mean': 1800.0,
                'std': 100.0
            }
        }
        
        # Normalized predictions (z-scores)
        normalized = np.array([-1.0, 0.0, 1.0])
        
        # Test
        denormalized = service.denormalize_predictions(normalized, scaling_params)
        
        # Verify
        assert np.allclose(denormalized, [1700.0, 1800.0, 1900.0])
    
    def test_denormalize_no_params(self):
        """Test denormalization without scaling params returns unchanged."""
        service = PredictionService()
        
        predictions = np.array([0.5, 0.6, 0.7])
        
        # Test with empty params
        result = service.denormalize_predictions(predictions, {})
        
        # Should return unchanged
        assert np.array_equal(result, predictions)
    
    def test_denormalize_uses_loaded_params(self):
        """Test denormalization uses loaded scaling params when not provided."""
        service = PredictionService()
        service.scaling_params = {
            'Close': {
                'method': 'minmax',
                'min': 1000.0,
                'max': 2000.0
            }
        }
        
        normalized = np.array([0.0, 0.5, 1.0])
        
        # Call without explicit params
        denormalized = service.denormalize_predictions(normalized)
        
        # Verify it used loaded params
        assert np.allclose(denormalized, [1000.0, 1500.0, 2000.0])


class TestPredictionIntervals:
    """Test prediction interval computation."""
    
    def test_compute_prediction_intervals_default_confidence(self):
        """Test computing prediction intervals with default confidence."""
        service = PredictionService()
        
        predictions = np.array([1800.0, 1850.0, 1900.0])
        
        # Test
        intervals = service.compute_prediction_intervals(predictions)
        
        # Verify
        assert intervals.shape == (3, 2)
        assert np.all(intervals[:, 0] < predictions)  # Lower bound < prediction
        assert np.all(intervals[:, 1] > predictions)  # Upper bound > prediction
    
    def test_compute_prediction_intervals_custom_confidence(self):
        """Test computing prediction intervals with custom confidence level."""
        service = PredictionService()
        
        predictions = np.array([1800.0, 1850.0])
        
        # Test with 90% confidence
        intervals = service.compute_prediction_intervals(predictions, confidence=0.90)
        
        # Verify shape
        assert intervals.shape == (2, 2)
    
    def test_prediction_intervals_increase_with_horizon(self):
        """Test that prediction intervals increase with forecast horizon."""
        service = PredictionService()
        
        predictions = np.array([1800.0, 1850.0, 1900.0, 1950.0, 2000.0])
        
        # Test
        intervals = service.compute_prediction_intervals(predictions)
        
        # Calculate interval widths
        widths = intervals[:, 1] - intervals[:, 0]
        
        # Widths should generally increase (uncertainty increases with horizon)
        assert widths[-1] >= widths[0]


class TestTimestampedPredictions:
    """Test timestamped prediction generation."""
    
    def test_predict_with_timestamps(self):
        """Test generating predictions with timestamps."""
        # Create service with mock model
        service = PredictionService()
        service.model = Mock()
        service.model.predict.side_effect = [
            np.array([[0.5]]),
            np.array([[0.52]]),
            np.array([[0.54]])
        ]
        service.model_type = 'LSTM'
        service.scaling_params = {
            'Close': {'method': 'minmax', 'min': 1800.0, 'max': 2000.0}
        }
        
        # Create input
        input_features = np.random.rand(60, 5)
        start_date = datetime(2024, 1, 1)
        
        # Test
        result = service.predict_with_timestamps(
            input_features, 
            start_date, 
            horizon=3,
            include_confidence=True
        )
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'Date' in result.columns
        assert 'Prediction' in result.columns
        assert 'Lower_CI' in result.columns
        assert 'Upper_CI' in result.columns
        
        # Check dates are business days
        assert result['Date'].iloc[0] >= pd.Timestamp(start_date)
    
    def test_predict_with_timestamps_no_confidence(self):
        """Test generating predictions without confidence intervals."""
        service = PredictionService()
        service.model = Mock()
        service.model.predict.side_effect = [
            np.array([[0.5]]),
            np.array([[0.52]])
        ]
        service.model_type = 'LSTM'
        service.scaling_params = {
            'Close': {'method': 'minmax', 'min': 1800.0, 'max': 2000.0}
        }
        
        input_features = np.random.rand(60, 5)
        start_date = datetime(2024, 1, 1)
        
        # Test
        result = service.predict_with_timestamps(
            input_features,
            start_date,
            horizon=2,
            include_confidence=False
        )
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'Date' in result.columns
        assert 'Prediction' in result.columns
        assert 'Lower_CI' not in result.columns
        assert 'Upper_CI' not in result.columns
    
    def test_predict_with_timestamps_denormalizes(self):
        """Test that timestamped predictions are denormalized."""
        service = PredictionService()
        service.model = Mock()
        service.model.predict.return_value = np.array([[0.5]])  # Normalized value
        service.model_type = 'LSTM'
        service.scaling_params = {
            'Close': {'method': 'minmax', 'min': 1000.0, 'max': 2000.0}
        }
        
        input_features = np.random.rand(60, 5)
        start_date = datetime(2024, 1, 1)
        
        # Test
        result = service.predict_with_timestamps(
            input_features,
            start_date,
            horizon=1,
            include_confidence=False
        )
        
        # Verify prediction is denormalized
        # 0.5 on [0,1] scale should map to 1500 on [1000, 2000] scale
        assert result['Prediction'].iloc[0] == pytest.approx(1500.0)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_step_with_1d_input_for_tree_model(self):
        """Test that 1D input works for tree-based models."""
        service = PredictionService()
        service.model = Mock()
        service.model.predict.return_value = np.array([0.5])
        service.model_type = 'RandomForest'
        
        # 1D input
        input_features = np.array([1.0, 2.0, 3.0])
        
        # Should work (expands to 2D internally)
        prediction = service.predict_single_step(input_features)
        
        assert isinstance(prediction, float)
    
    def test_multi_step_no_model_raises_error(self):
        """Test that multi-step prediction fails without loaded model."""
        service = PredictionService()
        
        with pytest.raises(ValueError, match="No model loaded"):
            service.predict_multi_step(np.array([1, 2, 3]), horizon=5)
    
    def test_denormalize_unknown_method(self):
        """Test denormalization with unknown method returns unchanged."""
        service = PredictionService()
        
        scaling_params = {
            'Close': {
                'method': 'unknown_method',
                'min': 1000.0,
                'max': 2000.0
            }
        }
        
        predictions = np.array([0.5, 0.6])
        
        result = service.denormalize_predictions(predictions, scaling_params)
        
        # Should return unchanged
        assert np.array_equal(result, predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
