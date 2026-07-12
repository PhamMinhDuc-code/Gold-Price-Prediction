"""
Quality Monitor Module

This module implements quality monitoring and prediction drift detection
for the Gold Price Prediction System.

Requirements: 10.5, 10.6
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from src.logger import get_logger
from src.exceptions import PredictionDriftError
from config import Config

logger = get_logger(__name__)


class QualityMonitor:
    """
    Monitor for data quality and prediction drift detection.
    
    This class provides functionality to:
    - Monitor data quality metrics
    - Detect prediction drift by comparing current metrics against baseline
    - Trigger alerts when quality degrades beyond threshold
    
    Requirements: 10.5, 10.6
    """
    
    def __init__(self, drift_threshold_pct: float = None):
        """
        Initialize QualityMonitor.
        
        Args:
            drift_threshold_pct: Percentage threshold for drift detection
                               (default: 25% from Config)
        
        Requirements: 10.5
        """
        self.drift_threshold_pct = (drift_threshold_pct if drift_threshold_pct is not None 
                                    else Config.DRIFT_THRESHOLD * 100)
        self.baseline_metrics = {}
        self.drift_alerts = []
        
        logger.info(f"QualityMonitor initialized with drift threshold: {self.drift_threshold_pct}%")
    
    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Set baseline performance metrics for drift detection.
        
        Args:
            metrics: Dictionary of metric names to values (e.g., MAE, RMSE, MAPE)
        
        Requirements: 10.5
        """
        self.baseline_metrics = metrics.copy()
        logger.info(f"Baseline metrics set: {metrics}")
    
    def load_baseline_from_test_set(self, model_metadata: Dict) -> None:
        """
        Load baseline metrics from model's test set performance.
        
        Args:
            model_metadata: Model metadata containing performance_metrics
        
        Requirements: 10.5
        """
        if 'performance_metrics' in model_metadata:
            self.set_baseline_metrics(model_metadata['performance_metrics'])
            logger.info("Baseline metrics loaded from model metadata")
        else:
            logger.warning("No performance_metrics found in model metadata")
    
    def detect_prediction_drift(self, current_metrics: Dict[str, float],
                               baseline_metrics: Optional[Dict[str, float]] = None,
                               raise_on_drift: bool = False) -> Tuple[bool, List[str]]:
        """
        Compare current metrics against baseline and detect drift.
        
        Checks if prediction error has increased beyond the threshold (default 25%).
        For error metrics (MAE, RMSE, MAPE), an increase indicates degradation.
        For quality metrics (R2, accuracy), a decrease indicates degradation.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline metrics to compare against.
                             If None, uses stored baseline.
            raise_on_drift: Whether to raise PredictionDriftError on detection
        
        Returns:
            Tuple of (drift_detected: bool, drift_messages: List[str])
        
        Raises:
            PredictionDriftError: If drift is detected and raise_on_drift=True
        
        Requirements: 10.5, 10.6
        """
        if baseline_metrics is None:
            baseline_metrics = self.baseline_metrics
        
        if not baseline_metrics:
            logger.warning("No baseline metrics available for drift detection")
            return False, []
        
        logger.info("Detecting prediction drift...")
        logger.info(f"Current metrics: {current_metrics}")
        logger.info(f"Baseline metrics: {baseline_metrics}")
        
        drift_detected = False
        drift_messages = []
        
        # Define which metrics are error metrics (higher = worse)
        error_metrics = {'mae', 'rmse', 'mape', 'loss'}
        
        # Define which metrics are quality metrics (higher = better)
        quality_metrics = {'r2', 'r_squared', 'accuracy', 'directional_accuracy'}
        
        # Check each metric present in both current and baseline
        for metric_name in current_metrics.keys():
            if metric_name not in baseline_metrics:
                logger.debug(f"Metric {metric_name} not in baseline, skipping")
                continue
            
            current_value = current_metrics[metric_name]
            baseline_value = baseline_metrics[metric_name]
            
            # Skip if baseline is zero (avoid division by zero)
            if baseline_value == 0:
                logger.warning(f"Baseline value for {metric_name} is zero, skipping")
                continue
            
            # Calculate percentage change
            pct_change = ((current_value - baseline_value) / abs(baseline_value)) * 100
            
            # Determine if this is degradation
            metric_lower = metric_name.lower()
            is_degradation = False
            
            if any(err_metric in metric_lower for err_metric in error_metrics):
                # For error metrics, increase = degradation
                is_degradation = pct_change > self.drift_threshold_pct
                degradation_pct = pct_change
            elif any(qual_metric in metric_lower for qual_metric in quality_metrics):
                # For quality metrics, decrease = degradation
                is_degradation = pct_change < -self.drift_threshold_pct
                degradation_pct = abs(pct_change)
            else:
                # Unknown metric type, treat as error metric
                logger.debug(f"Unknown metric type {metric_name}, treating as error metric")
                is_degradation = pct_change > self.drift_threshold_pct
                degradation_pct = pct_change
            
            if is_degradation:
                drift_detected = True
                drift_msg = (
                    f"Drift detected in {metric_name}: "
                    f"current={current_value:.4f}, baseline={baseline_value:.4f}, "
                    f"change={degradation_pct:.1f}% (threshold={self.drift_threshold_pct}%)"
                )
                drift_messages.append(drift_msg)
                logger.warning(drift_msg)
                
                # Store alert
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'baseline_value': baseline_value,
                    'degradation_pct': degradation_pct
                }
                self.drift_alerts.append(alert)
                
                # Raise exception if requested (Requirement 10.6)
                if raise_on_drift:
                    raise PredictionDriftError(
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        degradation_pct=degradation_pct,
                        threshold_pct=self.drift_threshold_pct
                    )
            else:
                logger.info(f"{metric_name}: change={pct_change:.1f}% (within threshold)")
        
        if drift_detected:
            logger.warning(f"Prediction drift detected in {len(drift_messages)} metric(s)")
            logger.warning("MODEL RETRAINING RECOMMENDED")
        else:
            logger.info("No prediction drift detected. Model performance is stable.")
        
        return drift_detected, drift_messages
    
    def get_drift_alerts(self) -> List[Dict]:
        """
        Get list of all drift alerts.
        
        Returns:
            List of drift alert dictionaries
        
        Requirements: 10.5
        """
        return self.drift_alerts.copy()
    
    def clear_drift_alerts(self) -> None:
        """
        Clear all stored drift alerts.
        
        Requirements: 10.5
        """
        self.drift_alerts.clear()
        logger.info("Drift alerts cleared")
    
    def generate_drift_report(self) -> Dict:
        """
        Generate comprehensive drift detection report.
        
        Returns:
            Dictionary containing drift report details
        
        Requirements: 10.5, 10.6
        """
        logger.info("Generating drift detection report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_metrics': self.baseline_metrics.copy(),
            'drift_threshold_pct': self.drift_threshold_pct,
            'total_alerts': len(self.drift_alerts),
            'alerts': self.drift_alerts.copy()
        }
        
        # Summarize alerts by metric
        alert_summary = {}
        for alert in self.drift_alerts:
            metric_name = alert['metric_name']
            if metric_name not in alert_summary:
                alert_summary[metric_name] = {
                    'count': 0,
                    'max_degradation': 0,
                    'avg_degradation': 0
                }
            
            alert_summary[metric_name]['count'] += 1
            alert_summary[metric_name]['max_degradation'] = max(
                alert_summary[metric_name]['max_degradation'],
                alert['degradation_pct']
            )
        
        # Calculate averages
        for metric_name, summary in alert_summary.items():
            metric_alerts = [a for a in self.drift_alerts if a['metric_name'] == metric_name]
            avg_degradation = sum(a['degradation_pct'] for a in metric_alerts) / len(metric_alerts)
            summary['avg_degradation'] = avg_degradation
        
        report['alert_summary'] = alert_summary
        
        logger.info(f"Drift report generated: {len(self.drift_alerts)} alerts")
        
        return report
    
    def check_data_quality_threshold(self, quality_score: float, 
                                    threshold: float = 70.0) -> bool:
        """
        Check if data quality score meets threshold.
        
        Args:
            quality_score: Data quality score (0-100)
            threshold: Minimum acceptable quality score
        
        Returns:
            True if quality meets threshold, False otherwise
        
        Requirements: 10.3
        """
        logger.info(f"Checking data quality: score={quality_score:.2f}, threshold={threshold}")
        
        meets_threshold = quality_score >= threshold
        
        if meets_threshold:
            logger.info("Data quality meets threshold")
        else:
            logger.warning(f"Data quality ({quality_score:.2f}) is below threshold ({threshold})")
            logger.warning("Dataset flagged as LOW QUALITY")
        
        return meets_threshold


if __name__ == "__main__":
    # Example usage and testing
    print("QualityMonitor module loaded successfully\n")
    
    # Initialize monitor
    monitor = QualityMonitor(drift_threshold_pct=25.0)
    print(f"Monitor initialized with {monitor.drift_threshold_pct}% threshold\n")
    
    # Set baseline metrics
    baseline = {
        'mae': 10.5,
        'rmse': 15.2,
        'mape': 0.05,
        'r2': 0.92
    }
    monitor.set_baseline_metrics(baseline)
    print(f"Baseline metrics: {baseline}\n")
    
    # Test 1: No drift (slight improvement)
    print("Test 1: No drift")
    current_good = {
        'mae': 10.0,
        'rmse': 14.8,
        'mape': 0.048,
        'r2': 0.93
    }
    drift, messages = monitor.detect_prediction_drift(current_good)
    print(f"Drift detected: {drift}")
    print(f"Messages: {messages}\n")
    
    # Test 2: Drift detected (error increased >25%)
    print("Test 2: Drift detected")
    current_bad = {
        'mae': 14.0,  # 33% increase
        'rmse': 20.0,  # 32% increase
        'mape': 0.05,
        'r2': 0.90
    }
    drift, messages = monitor.detect_prediction_drift(current_bad)
    print(f"Drift detected: {drift}")
    print(f"Messages: {messages}\n")
    
    # Generate report
    print("Drift Report:")
    report = monitor.generate_drift_report()
    print(f"Total alerts: {report['total_alerts']}")
    print(f"Alert summary: {report['alert_summary']}")
