"""
Custom Exception Classes for Gold Price Prediction System

This module defines custom exception classes for error handling throughout
the system.

Requirements: 10.1
"""


# ============================================================================
# Data Validation Exceptions
# ============================================================================

class DataValidationError(Exception):
    """Base exception for data validation errors."""
    pass


class MissingColumnError(DataValidationError):
    """Raised when required columns are missing from input data."""
    
    def __init__(self, missing_columns: list, message: str = None):
        self.missing_columns = missing_columns
        if message is None:
            message = f"Missing required columns: {', '.join(missing_columns)}"
        super().__init__(message)


class ChronologicalOrderError(DataValidationError):
    """Raised when dates are not in chronological order."""
    
    def __init__(self, first_violation_index: int = None, message: str = None):
        self.first_violation_index = first_violation_index
        if message is None:
            if first_violation_index is not None:
                message = f"Dates are not in chronological order (first violation at index {first_violation_index})"
            else:
                message = "Dates are not in chronological order"
        super().__init__(message)


class ConstraintViolationError(DataValidationError):
    """Raised when OHLC constraints are violated (e.g., High < Low)."""
    
    def __init__(self, constraint_type: str, violation_count: int, 
                 violation_indices: list = None, message: str = None):
        self.constraint_type = constraint_type
        self.violation_count = violation_count
        self.violation_indices = violation_indices or []
        
        if message is None:
            message = f"OHLC constraint violation: {constraint_type} in {violation_count} records"
            if violation_indices and len(violation_indices) > 0:
                sample_indices = violation_indices[:5]
                message += f" (first violations at indices: {sample_indices})"
        
        super().__init__(message)


# ============================================================================
# Prediction Exceptions
# ============================================================================

class PredictionError(Exception):
    """Base exception for prediction-related errors."""
    pass


class ModelLoadError(PredictionError):
    """Raised when model loading fails."""
    
    def __init__(self, model_path: str, reason: str, message: str = None):
        self.model_path = model_path
        self.reason = reason
        
        if message is None:
            message = f"Failed to load model from '{model_path}': {reason}"
        
        super().__init__(message)


class ExtrapolationWarning(UserWarning):
    """Warning for predictions outside training data range."""
    
    def __init__(self, feature_name: str, input_value: float, 
                 training_range: tuple, message: str = None):
        self.feature_name = feature_name
        self.input_value = input_value
        self.training_range = training_range
        
        if message is None:
            message = (f"Extrapolation warning: Feature '{feature_name}' value {input_value:.4f} "
                      f"is outside training range [{training_range[0]:.4f}, {training_range[1]:.4f}]")
        
        super().__init__(message)


# ============================================================================
# Data Quality Exceptions
# ============================================================================

class DataQualityError(Exception):
    """Raised when data quality is below acceptable threshold."""
    
    def __init__(self, quality_score: float, threshold: float = 70.0, 
                 issues: list = None, message: str = None):
        self.quality_score = quality_score
        self.threshold = threshold
        self.issues = issues or []
        
        if message is None:
            message = f"Data quality score ({quality_score:.2f}) is below threshold ({threshold})"
            if issues:
                message += f". Issues: {', '.join(issues)}"
        
        super().__init__(message)


class MissingValueThresholdError(DataQualityError):
    """Raised when missing values exceed acceptable threshold."""
    
    def __init__(self, feature_name: str, missing_pct: float, 
                 threshold_pct: float = 20.0, message: str = None):
        self.feature_name = feature_name
        self.missing_pct = missing_pct
        self.threshold_pct = threshold_pct
        
        if message is None:
            message = (f"Feature '{feature_name}' has {missing_pct:.1f}% missing values, "
                      f"exceeding threshold of {threshold_pct}%")
        
        super().__init__(message, missing_pct, threshold_pct, [f"{feature_name}: {missing_pct:.1f}%"])


# ============================================================================
# Model Quality Exceptions
# ============================================================================

class PredictionDriftError(Exception):
    """Raised when prediction quality has degraded significantly."""
    
    def __init__(self, metric_name: str, current_value: float, 
                 baseline_value: float, degradation_pct: float, 
                 threshold_pct: float = 25.0, message: str = None):
        self.metric_name = metric_name
        self.current_value = current_value
        self.baseline_value = baseline_value
        self.degradation_pct = degradation_pct
        self.threshold_pct = threshold_pct
        
        if message is None:
            message = (f"Prediction drift detected: {metric_name} increased by "
                      f"{degradation_pct:.1f}% (current: {current_value:.4f}, "
                      f"baseline: {baseline_value:.4f}), exceeding threshold of {threshold_pct}%")
        
        super().__init__(message)


# ============================================================================
# Utility Functions
# ============================================================================

def format_validation_errors(errors: list) -> str:
    """
    Format list of validation errors into readable message.
    
    Args:
        errors: List of error messages
    
    Returns:
        Formatted error message string
    """
    if not errors:
        return "No errors"
    
    if len(errors) == 1:
        return errors[0]
    
    formatted = f"Found {len(errors)} validation errors:\n"
    for i, error in enumerate(errors, 1):
        formatted += f"  {i}. {error}\n"
    
    return formatted.strip()


def format_validation_warnings(warnings: list) -> str:
    """
    Format list of validation warnings into readable message.
    
    Args:
        warnings: List of warning messages
    
    Returns:
        Formatted warning message string
    """
    if not warnings:
        return "No warnings"
    
    if len(warnings) == 1:
        return warnings[0]
    
    formatted = f"Found {len(warnings)} validation warnings:\n"
    for i, warning in enumerate(warnings, 1):
        formatted += f"  {i}. {warning}\n"
    
    return formatted.strip()


if __name__ == "__main__":
    # Test exception creation
    print("Testing custom exceptions...")
    
    # Test MissingColumnError
    try:
        raise MissingColumnError(['Open', 'Close'])
    except MissingColumnError as e:
        print(f"✓ MissingColumnError: {e}")
    
    # Test ChronologicalOrderError
    try:
        raise ChronologicalOrderError(first_violation_index=42)
    except ChronologicalOrderError as e:
        print(f"✓ ChronologicalOrderError: {e}")
    
    # Test ConstraintViolationError
    try:
        raise ConstraintViolationError('High < Low', 5, [10, 15, 20])
    except ConstraintViolationError as e:
        print(f"✓ ConstraintViolationError: {e}")
    
    # Test ModelLoadError
    try:
        raise ModelLoadError('/path/to/model', 'File not found')
    except ModelLoadError as e:
        print(f"✓ ModelLoadError: {e}")
    
    # Test ExtrapolationWarning
    import warnings
    warnings.simplefilter('always', ExtrapolationWarning)
    warnings.warn(ExtrapolationWarning('Close', 2500.0, (1000.0, 2000.0)))
    print("✓ ExtrapolationWarning created")
    
    # Test PredictionDriftError
    try:
        raise PredictionDriftError('RMSE', 150.0, 100.0, 50.0)
    except PredictionDriftError as e:
        print(f"✓ PredictionDriftError: {e}")
    
    print("\nAll exception tests passed!")
