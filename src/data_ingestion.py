"""
Data Ingestion and Validation Module

This module provides functionality to load and validate gold price OHLCV data
and economic indicators data.

Classes:
    - DataIngestionManager: Main class for data loading and validation
    - ValidationResult: Data class for validation results
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from src.logger import get_logger
from src.exceptions import (
    DataValidationError,
    MissingColumnError,
    ChronologicalOrderError,
    ConstraintViolationError
)

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Results from data validation checks."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataIngestionManager:
    """
    Manages data ingestion from CSV files and external APIs with validation.
    
    This class handles loading OHLCV data from CSV files, fetching economic
    indicators via yfinance API, and performing comprehensive validation checks.
    """
    
    def __init__(self, required_columns: Optional[List[str]] = None):
        """
        Initialize DataIngestionManager.
        
        Args:
            required_columns: List of required column names for OHLCV data.
                            Defaults to ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.required_columns = required_columns or ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        logger.info("DataIngestionManager initialized")
    
    def load_csv(self, filepath: str, date_column: str = 'Date', 
                 separator: str = ';') -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Reads CSV file containing OHLCV data, parses dates, and sets date as index.
        
        Args:
            filepath: Path to CSV file containing OHLCV data
            date_column: Name of the date column (default: 'Date')
            separator: CSV separator character (default: ';')
        
        Returns:
            DataFrame with OHLCV data indexed by date
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If date parsing fails
            
        Requirements: 1.1
        """
        try:
            logger.info(f"Loading CSV file from {filepath}")
            
            # Load CSV file
            df = pd.read_csv(filepath, sep=separator)
            
            # Parse dates - handle different date formats
            df[date_column] = pd.to_datetime(df[date_column], format='%Y.%m.%d %H:%M', errors='coerce')
            
            # Check for parsing failures
            if df[date_column].isna().any():
                failed_count = df[date_column].isna().sum()
                logger.warning(f"Failed to parse {failed_count} dates")
            
            # Set date as index
            df.set_index(date_column, inplace=True)
            df.index.name = 'Date'
            
            logger.info(f"Successfully loaded {len(df)} records from {filepath}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise ValueError(f"Failed to load CSV file: {str(e)}")
    
    def validate_ohlcv_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLCV data structure and constraints.
        
        Performs comprehensive validation including:
        - Required columns presence
        - High >= Low constraint
        - Close and Open within [Low, High] range
        
        Args:
            df: DataFrame containing OHLCV data
        
        Returns:
            ValidationResult with validation status and any errors/warnings
            
        Requirements: 1.2, 1.7, 1.8
        """
        errors = []
        warnings = []
        
        logger.info("Validating OHLCV data structure and constraints")
        
        # Check required columns (excluding Date since it's now the index)
        required_cols = [col for col in self.required_columns if col != 'Date']
        missing_columns = [col for col in required_cols if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            errors.append(error_msg)
            logger.error(error_msg)
            # Raise descriptive exception (Requirement 10.1)
            raise MissingColumnError(missing_columns)
        
        # Validate High >= Low constraint (Requirement 1.7)
        high_low_violations = df[df['High'] < df['Low']]
        if not high_low_violations.empty:
            violation_indices = high_low_violations.index.tolist()
            error_msg = f"High < Low constraint violated in {len(high_low_violations)} records"
            errors.append(error_msg)
            logger.error(f"{error_msg}: {violation_indices[:5]}")
            # Raise descriptive exception with specific details (Requirement 10.1)
            raise ConstraintViolationError(
                constraint_type='High < Low',
                violation_count=len(high_low_violations),
                violation_indices=violation_indices
            )
        
        # Validate Close within [Low, High] range (Requirement 1.8)
        close_violations = df[(df['Close'] < df['Low']) | (df['Close'] > df['High'])]
        if not close_violations.empty:
            violation_indices = close_violations.index.tolist()
            error_msg = f"Close outside [Low, High] range in {len(close_violations)} records"
            errors.append(error_msg)
            logger.error(f"{error_msg}: {violation_indices[:5]}")
            # Raise descriptive exception with specific details (Requirement 10.1)
            raise ConstraintViolationError(
                constraint_type='Close outside [Low, High]',
                violation_count=len(close_violations),
                violation_indices=violation_indices
            )
        
        # Validate Open within [Low, High] range (Requirement 1.8)
        open_violations = df[(df['Open'] < df['Low']) | (df['Open'] > df['High'])]
        if not open_violations.empty:
            violation_indices = open_violations.index.tolist()
            error_msg = f"Open outside [Low, High] range in {len(open_violations)} records"
            errors.append(error_msg)
            logger.error(f"{error_msg}: {violation_indices[:5]}")
            # Raise descriptive exception with specific details (Requirement 10.1)
            raise ConstraintViolationError(
                constraint_type='Open outside [Low, High]',
                violation_count=len(open_violations),
                violation_indices=violation_indices
            )
        
        # Check for missing values (flag for later handling)
        missing_values = df[required_cols].isna().sum()
        if missing_values.any():
            warning_msg = f"Missing values detected: {missing_values[missing_values > 0].to_dict()}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Check for negative values
        for col in ['Open', 'High', 'Low', 'Close']:
            if (df[col] < 0).any():
                warning_msg = f"Negative values detected in {col}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("OHLCV data validation passed")
        else:
            logger.error(f"OHLCV data validation failed with {len(errors)} errors")
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def validate_chronological_order(self, df: pd.DataFrame) -> bool:
        """
        Verify that dates are in chronological order.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            True if dates are in chronological order, False otherwise
            
        Requirements: 1.5
        """
        logger.info("Validating chronological order of dates")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index is not DatetimeIndex")
            return False
        
        # Check if index is sorted
        is_sorted = df.index.is_monotonic_increasing
        
        if is_sorted:
            logger.info("Dates are in chronological order")
        else:
            logger.error("Dates are NOT in chronological order")
            # Find first out-of-order date
            first_violation_idx = None
            for i in range(1, len(df.index)):
                if df.index[i] < df.index[i-1]:
                    first_violation_idx = i
                    logger.error(f"Out of order at index {i}: {df.index[i-1]} -> {df.index[i]}")
                    break
            # Raise descriptive exception (Requirement 10.1)
            raise ChronologicalOrderError(first_violation_index=first_violation_idx)
        
        return is_sorted
    
    def check_duplicates(self, df: pd.DataFrame) -> List[datetime]:
        """
        Identify duplicate date entries.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            List of duplicate dates
            
        Requirements: 1.6
        """
        logger.info("Checking for duplicate dates")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex")
            return []
        
        # Find duplicate dates
        duplicates = df.index[df.index.duplicated()].tolist()
        
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate dates: {duplicates[:5]}")
        else:
            logger.info("No duplicate dates found")
        
        return duplicates
    
    def load_economic_indicators(self, 
                                 tickers: Dict[str, str],
                                 start_date: str,
                                 end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch economic indicators using yfinance API.
        
        Downloads historical data for economic indicators (DXY, Oil, Treasury yields)
        and validates the data structure.
        
        Args:
            tickers: Dictionary mapping indicator names to yfinance ticker symbols
                    Example: {'DXY': 'DX-Y.NYB', 'Oil': 'CL=F', 'Treasury_10Y': '^TNX'}
            start_date: Start date for data download (format: 'YYYY-MM-DD')
            end_date: End date for data download (format: 'YYYY-MM-DD')
        
        Returns:
            Dictionary mapping indicator names to DataFrames with Close prices
        
        Raises:
            ValueError: If data download fails or validation errors occur
            
        Requirements: 1.3
        """
        logger.info(f"Fetching economic indicators from {start_date} to {end_date}")
        logger.info(f"Tickers: {tickers}")
        
        indicators_data = {}
        
        for name, ticker in tickers.items():
            try:
                logger.info(f"Downloading {name} ({ticker})")
                
                # Download data using yfinance
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    logger.error(f"No data downloaded for {name} ({ticker})")
                    raise ValueError(f"Failed to download data for {name} ({ticker})")
                
                # Extract Close prices and rename column
                indicator_df = data[['Close']].copy()
                indicator_df.columns = [name]
                
                # Validate data structure
                validation_result = self._validate_economic_indicator(indicator_df, name)
                
                if not validation_result.is_valid:
                    error_msg = f"Validation failed for {name}: {validation_result.errors}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                indicators_data[name] = indicator_df
                
                logger.info(f"Successfully loaded {name}: {len(indicator_df)} records")
                logger.info(f"{name} date range: {indicator_df.index.min()} to {indicator_df.index.max()}")
                
            except Exception as e:
                logger.error(f"Error downloading {name} ({ticker}): {str(e)}")
                raise ValueError(f"Failed to load economic indicator {name}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(indicators_data)} economic indicators")
        return indicators_data
    
    def _validate_economic_indicator(self, df: pd.DataFrame, name: str) -> ValidationResult:
        """
        Validate economic indicator data structure.
        
        Args:
            df: DataFrame containing indicator data
            name: Name of the indicator
        
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Check that we have a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append(f"{name}: Index is not DatetimeIndex")
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            warnings.append(f"{name}: Contains {missing_count} missing values")
        
        # Check that we have data
        if df.empty:
            errors.append(f"{name}: DataFrame is empty")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, List[datetime]]:
        """
        Flag missing values in OHLCV data and return affected records.
        
        Args:
            df: DataFrame containing OHLCV data with DatetimeIndex
        
        Returns:
            Dictionary mapping column names to list of dates with missing values
            
        Requirements: 1.4
        """
        logger.info("Detecting missing values in OHLCV data")
        
        missing_by_column = {}
        
        for column in df.columns:
            missing_mask = df[column].isna()
            if missing_mask.any():
                missing_dates = df[missing_mask].index.tolist()
                missing_by_column[column] = missing_dates
                logger.warning(f"{column}: {len(missing_dates)} missing values")
        
        if not missing_by_column:
            logger.info("No missing values detected")
        else:
            total_missing = sum(len(dates) for dates in missing_by_column.values())
            logger.warning(f"Total missing values across all columns: {total_missing}")
        
        return missing_by_column
    
    def get_validation_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive validation summary for OHLCV data.
        
        Args:
            df: DataFrame containing OHLCV data
        
        Returns:
            Dictionary containing validation results for all checks
        """
        logger.info("Generating comprehensive validation summary")
        
        summary = {
            'total_records': len(df),
            'date_range': (df.index.min(), df.index.max()) if not df.empty else (None, None),
            'ohlcv_validation': self.validate_ohlcv_data(df),
            'chronological_order': self.validate_chronological_order(df),
            'duplicates': self.check_duplicates(df),
            'missing_values': self.detect_missing_values(df)
        }
        
        return summary
