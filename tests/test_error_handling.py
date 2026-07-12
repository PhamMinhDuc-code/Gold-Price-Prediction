"""
Unit Tests for Error Handling and Monitoring

This module tests the error handling, validation, and monitoring functionality
of the Gold Price Prediction System.

Tests:
- Custom exception creation and behavior
- Data validation error handling
- Data quality threshold detection
- Extrapolation warnings
- Model load error handling
- Prediction drift detection

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from src.exceptions import (
    DataValidationError,
    MissingColumnError,
    ChronologicalOrderError,
    ConstraintViolationError,
    PredictionError,
    ModelLoadError,
    ExtrapolationWarning,
    DataQualityError,
    MissingValueThresholdError,
    PredictionDriftError,
    format_validation_errors,
    format_validation_warnings
)
from src.data_ingestion import DataIngestionManager
from src.data_preprocessing import DataPreprocessor
from src.quality_monitor import QualityMonitor

# Import PredictionService and ModelRegistry conditionally to avoid tensorflow import issues
try:
    from src.prediction_service import PredictionService
    from src.model_registry import ModelRegistry
    PREDICTION_SERVICE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PREDICTION_SERVICE_AVAILABLE = False
    PredictionService = None
    ModelRegistry = None


# ============================================================================
# Test Custom Exceptions (Requirement 10.1)
# ============================================================================

class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_missing_column_error(self):
        """Test MissingColumnError exception."""
        missing_cols = ['Open', 'Close']
        
        with pytest.raises(MissingColumnError) as exc_info:
            raise MissingColumnError(missing_cols)
        
        exception = exc_info.value
        assert exception.missing_columns == missing_cols
        assert 'Open' in str(exception)
        assert 'Close' in str(exception)
    
    def test_chronological_order_error(self):
        """Test ChronologicalOrderError exception."""
        with pytest.raises(ChronologicalOrderError) as exc_info:
            raise ChronologicalOrderError(first_violation_index=42)
        
        exception = exc_info.value
        assert exception.first_violation_index == 42
        assert '42' in str(exception)
    
    def test_constraint_violation_error(self):
        """Test ConstraintViolationError exception."""
        indices = [10, 15, 20, 25, 30]
        
        with pytest.raises(ConstraintViolationError) as exc_info:
            raise ConstraintViolationError('High < Low', 5, indices)
        
        exception = exc_info.value
        assert exception.constraint_type == 'High < Low'
        assert exception.violation_count == 5
        assert exception.violation_indices == indices
        assert 'High < Low' in str(exception)
        assert '5' in str(exception)
    
    def test_model_load_error(self):
        """Test ModelLoadError exception."""
        model_path = '/path/to/model'
        reason = 'File not found'
        
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError(model_path, reason)
        
        exception = exc_info.value
        assert exception.model_path == model_path
        assert exception.reason == reason
        assert model_path in str(exception)
        assert reason in str(exception)
    
    def test_extrapolation_warning(self):
        """Test ExtrapolationWarning creation."""
        feature_name = 'Close'
        input_value = 2500.0
        training_range = (1000.0, 2000.0)
        
        warning = ExtrapolationWarning(feature_name, input_value, training_range)
        
        assert warning.feature_name == feature_name
        assert warning.input_value == input_value
        assert warning.training_range == training_range
        assert 'Close' in str(warning)
        assert '2500.0' in str(warning)
    
    def test_missing_value_threshold_error(self):
        """Test MissingValueThresholdError exception."""
        with pytest.raises(MissingValueThresholdError) as exc_info:
            raise MissingValueThresholdError('Close', 25.0, 20.0)
        
        exception = exc_info.value
        assert exception.feature_name == 'Close'
        assert exception.missing_pct == 25.0
        assert exception.threshold_pct == 20.0
        assert 'Close' in str(exception)
        assert '25.0' in str(exception)
    
    def test_prediction_drift_error(self):
        """Test PredictionDriftError exception."""
        with pytest.raises(PredictionDriftError) as exc_info:
            raise PredictionDriftError('RMSE', 150.0, 100.0, 50.0, 25.0)
        
        exception = exc_info.value
        assert exception.metric_name == 'RMSE'
        assert exception.current_value == 150.0
        assert exception.baseline_value == 100.0
        assert exception.degradation_pct == 50.0
        assert exception.threshold_pct == 25.0
        assert 'RMSE' in str(exception)
        assert '50.0' in str(exception)


# ============================================================================
# Test Data Validation Error Handling (Requirement 10.1)
# ============================================================================

class TestDataValidationErrorHandling:
    """Test data validation error handling in DataIngestionManager."""
    
    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises MissingColumnError."""
        manager = DataIngestionManager()
        
        # Create DataFrame missing 'Close' column
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(MissingColumnError) as exc_info:
            manager.validate_ohlcv_data(df)
        
        assert 'Close' in exc_info.value.missing_columns
    
    def test_high_less_than_low_raises_error(self):
        """Test that High < Low raises ConstraintViolationError."""
        manager = DataIngestionManager()
        
        # Create DataFrame with High < Low violation
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 99, 108, 109],  # High < Low at index 2
            'Low': [99, 100, 101, 102, 103],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        
        with pytest.raises(ConstraintViolationError) as exc_info:
            manager.validate_ohlcv_data(df)
        
        assert exc_info.value.constraint_type == 'High < Low'
        assert exc_info.value.violation_count == 1
    
    def test_close_outside_range_raises_error(self):
        """Test that Close outside [Low, High] raises ConstraintViolationError."""
        manager = DataIngestionManager()
        
        # Create DataFrame with Close outside range
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [102, 103, 110, 105, 106],  # Close > High at index 2
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        
        with pytest.raises(ConstraintViolationError) as exc_info:
            manager.validate_ohlcv_data(df)
        
        assert 'Close outside' in exc_info.value.constraint_type
        assert exc_info.value.violation_count == 1
    
    def test_non_chronological_raises_error(self):
        """Test that non-chronological dates raise ChronologicalOrderError."""
        manager = DataIngestionManager()
        
        # Create DataFrame with non-chronological dates
        dates = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 4),  # Out of order
            datetime(2020, 1, 3),
            datetime(2020, 1, 5)
        ]
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.DatetimeIndex(dates))
        
        with pytest.raises(ChronologicalOrderError):
            manager.validate_chronological_order(df)


# ============================================================================
# Test Data Quality Monitoring (Requirement 10.3)
# ============================================================================

class TestDataQualityMonitoring:
    """Test data quality threshold detection."""
    
    def test_missing_values_exceed_threshold(self):
        """Test that missing values exceeding 20% threshold raises error."""
        preprocessor = DataPreprocessor(max_missing_pct=20.0)
        
        # Create DataFrame with >20% missing values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': [100.0] * 100,
            'Volume': [1000.0] * 100
        }, index=dates)
        
        # Add 25% missing values to Close column
        df.loc[df.index[:25], 'Close'] = np.nan
        
        with pytest.raises(MissingValueThresholdError) as exc_info:
            preprocessor.handle_missing_values(df, max_gap=3)
        
        exception = exc_info.value
        assert exception.feature_name == 'Close'
        assert exception.missing_pct == 25.0
        assert exception.threshold_pct == 20.0
    
    def test_missing_values_within_threshold(self):
        """Test that missing values within threshold are handled."""
        preprocessor = DataPreprocessor(max_missing_pct=20.0)
        
        # Create DataFrame with <20% missing values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': [100.0] * 100,
            'Volume': [1000.0] * 100
        }, index=dates)
        
        # Add 15% missing values (within threshold)
        df.loc[df.index[:15], 'Close'] = np.nan
        
        # Should not raise exception
        result = preprocessor.handle_missing_values(df, max_gap=3)
        assert result is not None
    
    def test_quality_monitor_threshold_check(self):
        """Test QualityMonitor data quality threshold checking."""
        monitor = QualityMonitor()
        
        # Test quality score below threshold
        assert monitor.check_data_quality_threshold(65.0, 70.0) == False
        
        # Test quality score at threshold
        assert monitor.check_data_quality_threshold(70.0, 70.0) == True
        
        # Test quality score above threshold
        assert monitor.check_data_quality_threshold(85.0, 70.0) == True


# ============================================================================
# Test Extrapolation Warnings (Requirement 10.2)
# ============================================================================

@pytest.mark.skipif(not PREDICTION_SERVICE_AVAILABLE, reason="PredictionService not available")
class TestExtrapolationWarnings:
    """Test extrapolation warning detection."""
    
    def test_extrapolation_warning_logged(self):
        """Test that extrapolation warnings are logged."""
        service = PredictionService()
        
        # Set up training data ranges
        service.training_data_range = {
            'Close': (1000.0, 2000.0),
            'Volume': (500.0, 1500.0)
        }
        service.feature_list = ['Close', 'Volume']
        
        # Create input features outside training range
        input_features = np.array([[2500.0, 1000.0]])  # Close is outside range
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ExtrapolationWarning)
            service.check_extrapolation(input_features)
            
            # Verify warning was issued
            assert len(w) > 0
            assert issubclass(w[0].category, ExtrapolationWarning)
            assert 'Close' in str(w[0].message)
            assert '2500.0' in str(w[0].message)
    
    def test_no_warning_within_range(self):
        """Test that no warning is issued when within training range."""
        service = PredictionService()
        
        # Set up training data ranges
        service.training_data_range = {
            'Close': (1000.0, 2000.0),
            'Volume': (500.0, 1500.0)
        }
        service.feature_list = ['Close', 'Volume']
        
        # Create input features within training range
        input_features = np.array([[1500.0, 1000.0]])
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ExtrapolationWarning)
            service.check_extrapolation(input_features)
            
            # Verify no warning was issued
            assert len(w) == 0


# ============================================================================
# Test Model Load Error Handling (Requirement 10.4)
# ============================================================================

@pytest.mark.skipif(not PREDICTION_SERVICE_AVAILABLE, reason="ModelRegistry not available")
class TestModelLoadErrorHandling:
    """Test model loading error handling."""
    
    def test_model_file_not_found_raises_error(self):
        """Test that missing model file raises ModelLoadError."""
        registry = ModelRegistry()
        
        # Try to load from non-existent path
        non_existent_path = Path('d:/nonexistent/model/path')
        
        with pytest.raises(ModelLoadError) as exc_info:
            registry.load_model(non_existent_path, 'LSTM')
        
        exception = exc_info.value
        assert 'nonexistent' in exception.model_path
        assert exception.reason is not None
    
    def test_metadata_file_not_found_raises_error(self):
        """Test that missing metadata file raises ModelLoadError."""
        registry = ModelRegistry()
        
        # Create temp directory without metadata file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            
            with pytest.raises(ModelLoadError) as exc_info:
                registry.load_metadata(model_path)
            
            exception = exc_info.value
            assert 'metadata' in exception.reason.lower()


# ============================================================================
# Test Prediction Drift Detection (Requirement 10.5, 10.6)
# ============================================================================

class TestPredictionDriftDetection:
    """Test prediction drift detection functionality."""
    
    def test_no_drift_detection(self):
        """Test that stable performance does not trigger drift."""
        monitor = QualityMonitor(drift_threshold_pct=25.0)
        
        # Set baseline metrics
        baseline = {
            'mae': 10.0,
            'rmse': 15.0,
            'mape': 0.05,
            'r2': 0.92
        }
        monitor.set_baseline_metrics(baseline)
        
        # Current metrics with slight improvement
        current = {
            'mae': 9.5,
            'rmse': 14.5,
            'mape': 0.048,
            'r2': 0.93
        }
        
        drift_detected, messages = monitor.detect_prediction_drift(current)
        
        assert drift_detected == False
        assert len(messages) == 0
    
    def test_drift_detected_error_increase(self):
        """Test that error increase >25% triggers drift detection."""
        monitor = QualityMonitor(drift_threshold_pct=25.0)
        
        # Set baseline metrics
        baseline = {
            'mae': 10.0,
            'rmse': 15.0
        }
        monitor.set_baseline_metrics(baseline)
        
        # Current metrics with >25% error increase
        current = {
            'mae': 14.0,  # 40% increase
            'rmse': 20.0  # 33% increase
        }
        
        drift_detected, messages = monitor.detect_prediction_drift(current)
        
        assert drift_detected == True
        assert len(messages) >= 1
        assert 'mae' in messages[0].lower() or 'rmse' in messages[0].lower()
    
    def test_drift_triggers_exception(self):
        """Test that drift detection can raise PredictionDriftError."""
        monitor = QualityMonitor(drift_threshold_pct=25.0)
        
        # Set baseline metrics
        baseline = {'mae': 10.0}
        monitor.set_baseline_metrics(baseline)
        
        # Current metrics with >25% error increase
        current = {'mae': 15.0}  # 50% increase
        
        with pytest.raises(PredictionDriftError) as exc_info:
            monitor.detect_prediction_drift(current, raise_on_drift=True)
        
        exception = exc_info.value
        assert exception.metric_name == 'mae'
        assert exception.degradation_pct > 25.0
    
    def test_drift_detection_quality_metrics(self):
        """Test drift detection for quality metrics (R2, accuracy)."""
        monitor = QualityMonitor(drift_threshold_pct=25.0)
        
        # Set baseline metrics
        baseline = {'r2': 0.90}
        monitor.set_baseline_metrics(baseline)
        
        # Current metrics with >25% decrease in R2
        current = {'r2': 0.65}  # 27.8% decrease
        
        drift_detected, messages = monitor.detect_prediction_drift(current)
        
        assert drift_detected == True
        assert len(messages) >= 1
        assert 'r2' in messages[0].lower()
    
    def test_get_drift_alerts(self):
        """Test retrieval of drift alerts."""
        monitor = QualityMonitor(drift_threshold_pct=25.0)
        
        baseline = {'mae': 10.0}
        monitor.set_baseline_metrics(baseline)
        
        # Trigger drift
        current = {'mae': 15.0}
        monitor.detect_prediction_drift(current)
        
        # Get alerts
        alerts = monitor.get_drift_alerts()
        assert len(alerts) == 1
        assert alerts[0]['metric_name'] == 'mae'
        assert 'timestamp' in alerts[0]
    
    def test_generate_drift_report(self):
        """Test drift report generation."""
        monitor = QualityMonitor(drift_threshold_pct=25.0)
        
        baseline = {'mae': 10.0, 'rmse': 15.0}
        monitor.set_baseline_metrics(baseline)
        
        # Trigger drift
        current = {'mae': 15.0, 'rmse': 20.0}
        monitor.detect_prediction_drift(current)
        
        # Generate report
        report = monitor.generate_drift_report()
        
        assert 'timestamp' in report
        assert 'baseline_metrics' in report
        assert 'total_alerts' in report
        assert report['total_alerts'] == 2
        assert 'alert_summary' in report


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions for error formatting."""
    
    def test_format_validation_errors_single(self):
        """Test formatting a single error message."""
        errors = ["Missing column: Close"]
        result = format_validation_errors(errors)
        assert result == "Missing column: Close"
    
    def test_format_validation_errors_multiple(self):
        """Test formatting multiple error messages."""
        errors = ["Missing column: Close", "High < Low violation", "Duplicate dates"]
        result = format_validation_errors(errors)
        assert "3 validation errors" in result
        assert "1." in result
        assert "2." in result
        assert "3." in result
    
    def test_format_validation_errors_empty(self):
        """Test formatting empty error list."""
        errors = []
        result = format_validation_errors(errors)
        assert result == "No errors"
    
    def test_format_validation_warnings(self):
        """Test formatting warning messages."""
        warnings = ["Missing values detected", "Negative values in Volume"]
        result = format_validation_warnings(warnings)
        assert "2 validation warnings" in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""
    
    def test_end_to_end_validation_error_handling(self):
        """Test complete validation workflow with error handling."""
        manager = DataIngestionManager()
        
        # Create invalid DataFrame
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [99] * 10,
            'Close': [110] * 10,  # Close > High (violation)
            'Volume': [1000] * 10
        }, index=dates)
        
        # Should raise ConstraintViolationError with descriptive message
        with pytest.raises(ConstraintViolationError) as exc_info:
            manager.validate_ohlcv_data(df)
        
        exception = exc_info.value
        assert exception.violation_count == 10
        assert len(exception.violation_indices) > 0
        assert 'Close' in str(exception)
    
    def test_end_to_end_quality_monitoring(self):
        """Test complete quality monitoring workflow."""
        # Create preprocessor and quality monitor
        preprocessor = DataPreprocessor(max_missing_pct=20.0)
        monitor = QualityMonitor(drift_threshold_pct=25.0)
        
        # Set up drift detection
        baseline_metrics = {'mae': 10.0, 'rmse': 15.0}
        monitor.set_baseline_metrics(baseline_metrics)
        
        # Simulate performance degradation
        current_metrics = {'mae': 14.0, 'rmse': 20.0}
        drift_detected, messages = monitor.detect_prediction_drift(current_metrics)
        
        assert drift_detected == True
        assert len(messages) >= 1
        
        # Generate report
        report = monitor.generate_drift_report()
        assert report['total_alerts'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
