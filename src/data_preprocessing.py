"""
Data Preprocessing and Cleaning Module

This module provides functionality to clean, normalize, and prepare data for model training.

Classes:
    - DataPreprocessor: Main class for data preprocessing and cleaning
    - DataQualityReport: Data class for quality reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from src.logger import get_logger
from src.exceptions import MissingValueThresholdError, DataQualityError
from config import Config

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """Report containing data quality metrics."""
    total_records: int
    missing_values_handled: Dict[str, int]
    outliers_removed: Dict[str, int]
    date_range: Tuple[datetime, datetime]
    data_quality_score: float  # 0-100


class DataPreprocessor:
    """
    Handles data preprocessing and cleaning operations.
    
    This class provides methods for handling missing values, normalization,
    dataset alignment, outlier removal, and quality reporting.
    """
    
    def __init__(self, max_forward_fill_gap: int = None, 
                 outlier_std_threshold: float = None,
                 max_missing_pct: float = None):
        """
        Initialize DataPreprocessor with configuration parameters.
        
        Args:
            max_forward_fill_gap: Maximum gap days for forward-fill (default: from Config)
            outlier_std_threshold: Number of std deviations for outlier detection (default: from Config)
            max_missing_pct: Maximum allowed missing value percentage (default: from Config)
        """
        self.max_forward_fill_gap = max_forward_fill_gap or Config.MAX_FORWARD_FILL_GAP
        self.outlier_std_threshold = outlier_std_threshold or Config.OUTLIER_STD_THRESHOLD
        self.max_missing_pct = max_missing_pct or Config.MAX_MISSING_PCT
        
        # Track preprocessing statistics
        self.stats = {
            'missing_values_handled': {},
            'outliers_removed': {},
            'original_record_count': 0,
            'final_record_count': 0
        }
        
        logger.info("DataPreprocessor initialized")
        logger.info(f"Max forward-fill gap: {self.max_forward_fill_gap} days")
        logger.info(f"Outlier threshold: {self.outlier_std_threshold} std deviations")
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             max_gap: Optional[int] = None) -> pd.DataFrame:
        """
        Handle missing values using forward-fill for small gaps.
        
        Applies forward-fill interpolation for gaps of specified days or fewer.
        For gaps larger than max_gap, leaves values as NaN for later handling.
        
        Args:
            df: DataFrame with potential missing values
            max_gap: Maximum gap days for forward-fill (default: uses instance setting)
        
        Returns:
            DataFrame with missing values handled
            
        Requirements: 2.1
        """
        if max_gap is None:
            max_gap = self.max_forward_fill_gap
        
        logger.info(f"Handling missing values with forward-fill (max gap: {max_gap} days)")
        
        df_copy = df.copy()
        missing_before = df_copy.isna().sum()
        
        # Check if missing values exceed threshold (Requirement 10.3)
        total_records = len(df_copy)
        for column in df_copy.columns:
            if missing_before[column] > 0:
                missing_pct = (missing_before[column] / total_records) * 100
                logger.info(f"{column}: {missing_before[column]} missing values ({missing_pct:.1f}%)")
                
                # Flag dataset as low quality if missing values exceed 20% threshold
                if missing_pct > self.max_missing_pct:
                    logger.warning(f"{column}: Missing value percentage ({missing_pct:.1f}%) exceeds threshold ({self.max_missing_pct}%)")
                    logger.warning(f"Dataset flagged as LOW QUALITY for column: {column}")
                    # Raise exception (Requirement 10.3)
                    raise MissingValueThresholdError(
                        feature_name=column,
                        missing_pct=missing_pct,
                        threshold_pct=self.max_missing_pct
                    )
        
        # Apply forward-fill with limit
        df_filled = df_copy.ffill(limit=max_gap)
        
        # Calculate how many values were filled
        missing_after = df_filled.isna().sum()
        filled_count = missing_before - missing_after
        
        # Update statistics
        for column in df_copy.columns:
            if filled_count[column] > 0:
                self.stats['missing_values_handled'][column] = int(filled_count[column])
                logger.info(f"{column}: Filled {filled_count[column]} missing values")
        
        # Warn about remaining missing values
        remaining_missing = missing_after.sum()
        if remaining_missing > 0:
            logger.warning(f"Remaining missing values after forward-fill: {remaining_missing}")
            logger.warning(f"These gaps exceed {max_gap} days and require manual handling")
        
        return df_filled
    
    def interpolate_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply linear interpolation for economic indicators.
        
        Uses linear interpolation for continuous variables to fill missing values.
        
        Args:
            df: DataFrame containing economic indicator data
        
        Returns:
            DataFrame with interpolated values
            
        Requirements: 2.2
        """
        logger.info("Applying linear interpolation for economic indicators")
        
        df_copy = df.copy()
        missing_before = df_copy.isna().sum()
        
        # Track missing values by column
        for column in df_copy.columns:
            if missing_before[column] > 0:
                logger.info(f"{column}: {missing_before[column]} missing values before interpolation")
        
        # Apply linear interpolation
        df_interpolated = df_copy.interpolate(method='linear', limit_direction='both')
        
        # Calculate interpolated values
        missing_after = df_interpolated.isna().sum()
        interpolated_count = missing_before - missing_after
        
        # Update statistics
        for column in df_copy.columns:
            if interpolated_count[column] > 0:
                if column in self.stats['missing_values_handled']:
                    self.stats['missing_values_handled'][column] += int(interpolated_count[column])
                else:
                    self.stats['missing_values_handled'][column] = int(interpolated_count[column])
                logger.info(f"{column}: Interpolated {interpolated_count[column]} missing values")
        
        # Warn about any remaining missing values
        remaining_missing = missing_after.sum()
        if remaining_missing > 0:
            logger.warning(f"Remaining missing values after interpolation: {remaining_missing}")
        
        return df_interpolated
    
    def normalize_features(self, df: pd.DataFrame, 
                          method: str = None,
                          exclude_columns: Optional[list] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize numerical features and return scaling parameters.
        
        Supports min-max normalization (scales to [0, 1]) and z-score standardization.
        Stores scaling parameters for later inverse transformation.
        
        Args:
            df: DataFrame with features to normalize
            method: Normalization method ('minmax' or 'zscore', default: from Config)
            exclude_columns: List of columns to exclude from normalization
        
        Returns:
            Tuple of (normalized_df, scaling_params_dict)
            
        Requirements: 2.3
        """
        if method is None:
            method = Config.NORMALIZATION_METHOD
        
        logger.info(f"Normalizing features using {method} method")
        
        df_copy = df.copy()
        scaling_params = {
            'method': method,
            'params': {}
        }
        
        exclude_columns = exclude_columns or []
        columns_to_normalize = [col for col in df_copy.columns if col not in exclude_columns]
        
        logger.info(f"Normalizing {len(columns_to_normalize)} columns")
        
        if method == 'minmax':
            # Min-Max normalization: (x - min) / (max - min)
            for column in columns_to_normalize:
                col_min = df_copy[column].min()
                col_max = df_copy[column].max()
                
                # Store parameters
                scaling_params['params'][column] = {
                    'min': float(col_min),
                    'max': float(col_max)
                }
                
                # Apply normalization
                if col_max > col_min:
                    df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
                    logger.debug(f"{column}: Normalized to [{df_copy[column].min():.4f}, {df_copy[column].max():.4f}]")
                else:
                    logger.warning(f"{column}: No variation (min=max), skipping normalization")
                    df_copy[column] = 0.0
        
        elif method == 'zscore':
            # Z-score standardization: (x - mean) / std
            for column in columns_to_normalize:
                col_mean = df_copy[column].mean()
                col_std = df_copy[column].std()
                
                # Store parameters
                scaling_params['params'][column] = {
                    'mean': float(col_mean),
                    'std': float(col_std)
                }
                
                # Apply standardization
                if col_std > 0:
                    df_copy[column] = (df_copy[column] - col_mean) / col_std
                    logger.debug(f"{column}: Standardized (mean={df_copy[column].mean():.4f}, std={df_copy[column].std():.4f})")
                else:
                    logger.warning(f"{column}: No variation (std=0), skipping standardization")
                    df_copy[column] = 0.0
        
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'zscore'.")
        
        logger.info(f"Successfully normalized {len(columns_to_normalize)} columns")
        
        return df_copy, scaling_params
    
    def denormalize_features(self, df: pd.DataFrame, 
                            scaling_params: Dict) -> pd.DataFrame:
        """
        Denormalize features using stored scaling parameters.
        
        Performs inverse transformation to restore original scale.
        
        Args:
            df: DataFrame with normalized features
            scaling_params: Dictionary containing scaling parameters
        
        Returns:
            DataFrame with denormalized values
        """
        logger.info("Denormalizing features")
        
        df_copy = df.copy()
        method = scaling_params['method']
        params = scaling_params['params']
        
        if method == 'minmax':
            for column, col_params in params.items():
                if column in df_copy.columns:
                    col_min = col_params['min']
                    col_max = col_params['max']
                    df_copy[column] = df_copy[column] * (col_max - col_min) + col_min
        
        elif method == 'zscore':
            for column, col_params in params.items():
                if column in df_copy.columns:
                    col_mean = col_params['mean']
                    col_std = col_params['std']
                    df_copy[column] = df_copy[column] * col_std + col_mean
        
        logger.info("Denormalization complete")
        
        return df_copy
    
    def align_datasets(self, gold_df: pd.DataFrame, 
                       indicators: Dict[str, pd.DataFrame],
                       strategy: str = 'inner') -> pd.DataFrame:
        """
        Align multiple datasets by date using join strategy.
        
        Merges gold data with economic indicators using date-based join.
        Supports inner join and forward-fill strategy.
        
        Args:
            gold_df: DataFrame with gold OHLCV data
            indicators: Dictionary mapping indicator names to DataFrames
            strategy: Join strategy ('inner' or 'forward_fill')
        
        Returns:
            Aligned DataFrame with all datasets merged by date
            
        Requirements: 2.4
        """
        logger.info(f"Aligning datasets using {strategy} strategy")
        logger.info(f"Gold data: {len(gold_df)} records")
        
        # Start with gold data
        aligned_df = gold_df.copy()
        
        # Join each indicator
        for name, indicator_df in indicators.items():
            logger.info(f"Joining {name}: {len(indicator_df)} records")
            
            if strategy == 'inner':
                # Inner join - keep only matching dates
                aligned_df = aligned_df.join(indicator_df, how='inner')
            elif strategy == 'forward_fill':
                # Outer join with forward-fill
                aligned_df = aligned_df.join(indicator_df, how='left')
                # Forward-fill missing indicator values
                aligned_df[indicator_df.columns] = aligned_df[indicator_df.columns].ffill()
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'inner' or 'forward_fill'.")
        
        logger.info(f"Aligned dataset: {len(aligned_df)} records")
        logger.info(f"Date range: {aligned_df.index.min()} to {aligned_df.index.max()}")
        logger.info(f"Columns: {list(aligned_df.columns)}")
        
        return aligned_df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       n_std: Optional[float] = None,
                       columns: Optional[list] = None) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        
        Removes records where any feature falls beyond n standard deviations from mean.
        Logs outlier statistics before removal.
        
        Args:
            df: DataFrame with potential outliers
            n_std: Number of standard deviations threshold (default: uses instance setting)
            columns: List of columns to check for outliers (default: all numeric columns)
        
        Returns:
            DataFrame with outliers removed
            
        Requirements: 2.5
        """
        if n_std is None:
            n_std = self.outlier_std_threshold
        
        logger.info(f"Removing outliers using z-score method (threshold: {n_std} std)")
        
        df_copy = df.copy()
        original_count = len(df_copy)
        
        # Determine columns to check
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Checking {len(columns)} columns for outliers")
        
        # Calculate z-scores for each column
        outlier_mask = pd.Series([False] * len(df_copy), index=df_copy.index)
        
        for column in columns:
            col_mean = df_copy[column].mean()
            col_std = df_copy[column].std()
            
            if col_std > 0:
                z_scores = np.abs((df_copy[column] - col_mean) / col_std)
                column_outliers = z_scores > n_std
                
                if column_outliers.any():
                    outlier_count = column_outliers.sum()
                    self.stats['outliers_removed'][column] = int(outlier_count)
                    logger.info(f"{column}: {outlier_count} outliers detected")
                    
                    # Log some outlier values for inspection
                    outlier_values = df_copy[column_outliers][column]
                    logger.debug(f"{column} outlier range: [{outlier_values.min():.4f}, {outlier_values.max():.4f}]")
                
                # Update mask
                outlier_mask = outlier_mask | column_outliers
        
        # Remove outliers
        df_cleaned = df_copy[~outlier_mask]
        removed_count = original_count - len(df_cleaned)
        
        logger.info(f"Removed {removed_count} records containing outliers ({removed_count/original_count*100:.2f}%)")
        logger.info(f"Remaining records: {len(df_cleaned)}")
        
        return df_cleaned
    
    def generate_quality_report(self, df: pd.DataFrame, 
                               original_record_count: Optional[int] = None) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Calculates and returns records processed, missing values handled, outliers removed,
        and overall data quality score (0-100).
        
        Args:
            df: Final processed DataFrame
            original_record_count: Original number of records before preprocessing
        
        Returns:
            DataQualityReport with quality metrics
            
        Requirements: 2.6
        """
        logger.info("Generating data quality report")
        
        if original_record_count is None:
            original_record_count = self.stats.get('original_record_count', len(df))
        
        # Calculate date range
        date_range = (df.index.min(), df.index.max()) if len(df) > 0 else (None, None)
        
        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(df, original_record_count)
        
        report = DataQualityReport(
            total_records=len(df),
            missing_values_handled=self.stats['missing_values_handled'].copy(),
            outliers_removed=self.stats['outliers_removed'].copy(),
            date_range=date_range,
            data_quality_score=quality_score
        )
        
        # Log report summary
        logger.info("=" * 60)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 60)
        logger.info(f"Total records: {report.total_records}")
        logger.info(f"Date range: {report.date_range[0]} to {report.date_range[1]}")
        logger.info(f"Missing values handled: {sum(report.missing_values_handled.values())}")
        logger.info(f"Outliers removed: {sum(report.outliers_removed.values())}")
        logger.info(f"Data quality score: {report.data_quality_score:.2f}/100")
        logger.info("=" * 60)
        
        # Check quality threshold
        if quality_score < 70:
            logger.warning(f"Data quality score ({quality_score:.2f}) is below recommended threshold (70)")
        
        return report
    
    def _calculate_quality_score(self, df: pd.DataFrame, 
                                 original_record_count: int) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Considers:
        - Completeness (remaining records vs original)
        - Missing value handling effectiveness
        - Outlier removal proportion
        - Final data cleanliness
        
        Args:
            df: Final processed DataFrame
            original_record_count: Original number of records
        
        Returns:
            Quality score (0-100)
        """
        score_components = []
        
        # 1. Completeness score (30 points): percentage of records retained
        if original_record_count > 0:
            retention_rate = len(df) / original_record_count
            completeness_score = min(retention_rate * 30, 30)
            score_components.append(completeness_score)
        else:
            score_components.append(0)
        
        # 2. Missing values score (30 points): no missing values = full points
        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            missing_cells = df.isna().sum().sum()
            missing_rate = missing_cells / total_cells
            missing_score = max(0, 30 * (1 - missing_rate * 5))  # Penalize heavily
            score_components.append(missing_score)
        else:
            score_components.append(0)
        
        # 3. Outlier handling score (20 points): reasonable outlier removal
        total_outliers_removed = sum(self.stats['outliers_removed'].values())
        if original_record_count > 0:
            outlier_rate = total_outliers_removed / original_record_count
            # Ideal outlier rate: 1-5% removed
            if 0.01 <= outlier_rate <= 0.05:
                outlier_score = 20
            elif outlier_rate < 0.01:
                outlier_score = 15  # Maybe too lenient
            else:
                outlier_score = max(0, 20 * (1 - (outlier_rate - 0.05) * 2))
            score_components.append(outlier_score)
        else:
            score_components.append(20)
        
        # 4. Data validity score (20 points): no negative values in price columns
        validity_score = 20
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                if (df[col] < 0).any():
                    validity_score -= 5
        score_components.append(max(0, validity_score))
        
        # Calculate total score
        total_score = sum(score_components)
        
        logger.debug(f"Quality score components: {score_components}")
        logger.debug(f"Total quality score: {total_score:.2f}")
        
        return round(total_score, 2)
    
    def reset_statistics(self):
        """Reset preprocessing statistics for new preprocessing run."""
        self.stats = {
            'missing_values_handled': {},
            'outliers_removed': {},
            'original_record_count': 0,
            'final_record_count': 0
        }
        logger.info("Preprocessing statistics reset")
