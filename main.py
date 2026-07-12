"""
Gold Price Prediction System - Main Pipeline Orchestration

This script orchestrates the end-to-end pipeline for gold price prediction including:
- Data ingestion and preprocessing
- Feature engineering
- Model training
- Prediction generation
- Model evaluation

Usage:
    Training mode:
        python main.py train --data data/gold_data.csv --model-type LSTM --version v1.0
    
    Prediction mode:
        python main.py predict --model-version v1.0 --horizon 30 --output predictions.csv
    
    Evaluation mode:
        python main.py evaluate --model-version v1.0 --test-data data/test_data.csv

Requirements: All requirements (integration)
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import json

from config import Config
from src.data_ingestion import DataIngestionManager
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.dataset_splitter import DatasetSplitter

# Lazy import for model training to avoid TensorFlow dependency on import
# Will be imported when needed
ModelTrainingPipeline = None
ModelMetadata = None

from src.prediction_service import PredictionService
from src.model_evaluator import ModelEvaluator
from src.visualization_manager import VisualizationManager
from src.model_registry import ModelRegistry
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class GoldPricePipeline:
    """
    Main pipeline orchestrator for gold price prediction system.
    
    This class coordinates all pipeline stages and modes.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object (uses default Config if None)
        """
        self.config = config if config is not None else Config
        
        # Initialize components
        self.data_ingestion = DataIngestionManager()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.dataset_splitter = DatasetSplitter()
        self.training_pipeline = None  # Lazy initialization
        self.prediction_service = PredictionService()
        self.evaluator = ModelEvaluator()
        self.visualizer = VisualizationManager()
        self.model_registry = ModelRegistry()
        
        logger.info("GoldPricePipeline initialized")
    
    def run_training_pipeline(self, data_path: str, model_type: str,
                             version: str, indicators_config: Optional[Dict] = None,
                             hyperparams: Optional[Dict] = None,
                             tune_hyperparams: bool = False) -> str:
        """
        Execute full training pipeline.
        
        Pipeline stages:
        1. Load data
        2. Preprocess data
        3. Engineer features
        4. Split dataset
        5. Train model
        6. Evaluate model
        7. Save model with metadata
        8. Generate training report
        
        Args:
            data_path: Path to gold price CSV file
            model_type: Type of model ('LSTM', 'GRU', 'XGBoost', 'RandomForest')
            version: Version string for the model
            indicators_config: Optional config for economic indicators
            hyperparams: Optional hyperparameter dictionary
            tune_hyperparams: Whether to perform hyperparameter tuning
        
        Returns:
            Path to saved model directory
        
        Requirements: 1.1-5.6, 7.1-7.7
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Data path: {data_path}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Version: {version}")
        
        # Lazy import model training components
        global ModelTrainingPipeline, ModelMetadata
        if ModelTrainingPipeline is None:
            from src.model_training import ModelTrainingPipeline, ModelMetadata
        
        # Initialize training pipeline if needed
        if self.training_pipeline is None:
            self.training_pipeline = ModelTrainingPipeline(config=self.config)
        
        try:
            # Stage 1: Load data
            logger.info("\n[1/8] Loading data...")
            df_gold = self.data_ingestion.load_csv(data_path)
            self.data_ingestion.validate_ohlcv_data(df_gold)
            self.data_ingestion.validate_chronological_order(df_gold)
            
            # Load economic indicators if config provided
            if indicators_config:
                logger.info("Loading economic indicators...")
                start_date = df_gold.index.min().strftime('%Y-%m-%d')
                end_date = df_gold.index.max().strftime('%Y-%m-%d')
                indicators = self.data_ingestion.load_economic_indicators(
                    tickers=indicators_config,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                indicators = {}
            
            # Stage 2: Preprocess data
            logger.info("\n[2/8] Preprocessing data...")
            df_gold = self.preprocessor.handle_missing_values(df_gold)
            
            if indicators:
                for name, indicator_df in indicators.items():
                    indicators[name] = self.preprocessor.interpolate_economic_indicators(indicator_df)
                
                # Align datasets
                df_combined = self.preprocessor.align_datasets(df_gold, indicators, strategy='inner')
            else:
                df_combined = df_gold
            
            # Remove outliers
            df_combined = self.preprocessor.remove_outliers(df_combined)
            
            # Stage 3: Engineer features
            logger.info("\n[3/8] Engineering features...")
            df_features = self.feature_engineer.build_feature_set(
                df_combined,
                create_lags=True,
                create_rolling=True,
                create_technical=True,
                create_interactions=True,
                create_temporal=True,
                handle_nan=True
            )
            
            # Store feature list
            feature_list = [col for col in df_features.columns if col != self.config.TARGET_COLUMN]
            
            # Normalize features
            df_normalized, scaling_params = self.preprocessor.normalize_features(
                df_features,
                method=self.config.NORMALIZATION_METHOD
            )
            
            # Stage 4: Split dataset
            logger.info("\n[4/8] Splitting dataset...")
            train_df, val_df, test_df = self.dataset_splitter.split_dataset(df_normalized)
            self.dataset_splitter.verify_split_integrity(train_df, val_df, test_df)
            
            # Prepare features and targets
            if model_type in ['LSTM', 'GRU']:
                # Sequence models - create sequences
                X_train, y_train = self.dataset_splitter.prepare_feature_target_split(train_df)
                X_val, y_val = self.dataset_splitter.prepare_feature_target_split(val_df)
                X_test, y_test = self.dataset_splitter.prepare_feature_target_split(test_df)
                
                # Create sequences
                X_train, y_train = self.dataset_splitter.create_sequences(X_train, y_train)
                X_val, y_val = self.dataset_splitter.create_sequences(X_val, y_val)
                X_test, y_test = self.dataset_splitter.create_sequences(X_test, y_test)
                
                sequence_length = self.config.SEQUENCE_LENGTH
            else:
                # Tree-based models - no sequences
                X_train, y_train = self.dataset_splitter.prepare_feature_target_split(train_df)
                X_val, y_val = self.dataset_splitter.prepare_feature_target_split(val_df)
                X_test, y_test = self.dataset_splitter.prepare_feature_target_split(test_df)
                
                sequence_length = None
            
            logger.info(f"Training set: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Validation set: X={X_val.shape}, y={y_val.shape}")
            logger.info(f"Test set: X={X_test.shape}, y={y_test.shape}")
            
            # Stage 5: Train model
            logger.info("\n[5/8] Training model...")
            
            if tune_hyperparams:
                logger.info("Performing hyperparameter tuning...")
                tuning_result = self.training_pipeline.hyperparameter_tuning(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    search_space=hyperparams
                )
                
                best_hyperparams = tuning_result['best_params']
                model = tuning_result['best_model']
                logger.info(f"Best hyperparameters: {best_hyperparams}")
                
                # For deep learning, training already done during tuning
                if model_type in ['LSTM', 'GRU']:
                    training_result = None  # Already trained
                else:
                    training_result = None  # Already trained
            else:
                # Build model with provided or default hyperparams
                input_shape = (X_train.shape[1], X_train.shape[2]) if model_type in ['LSTM', 'GRU'] else None
                
                if model_type == 'LSTM':
                    model = self.training_pipeline.build_lstm_model(input_shape, hyperparams)
                elif model_type == 'GRU':
                    model = self.training_pipeline.build_gru_model(input_shape, hyperparams)
                elif model_type == 'XGBoost':
                    model = self.training_pipeline.build_xgboost_model(hyperparams)
                elif model_type == 'RandomForest':
                    model = self.training_pipeline.build_random_forest_model(hyperparams)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                # Train model
                model_type_category = 'deep_learning' if model_type in ['LSTM', 'GRU'] else 'tree_based'
                checkpoint_path = self.config.MODEL_DIR / 'checkpoints' / f'{model_type.lower()}_best.keras'
                
                training_result = self.training_pipeline.train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_type=model_type_category,
                    checkpoint_path=str(checkpoint_path) if model_type in ['LSTM', 'GRU'] else None
                )
                
                best_hyperparams = hyperparams or {}
            
            # Stage 6: Evaluate model
            logger.info("\n[6/8] Evaluating model...")
            
            # Generate predictions on test set
            if model_type in ['LSTM', 'GRU']:
                y_pred = model.predict(X_test, verbose=0).flatten()
            else:
                # Flatten for tree-based models if needed
                X_test_flat = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) == 3 else X_test
                y_pred = model.predict(X_test_flat)
            
            # Denormalize predictions and actuals
            target_params = scaling_params['params'][self.config.TARGET_COLUMN]
            if scaling_params['method'] == 'minmax':
                y_test_original = y_test * (target_params['max'] - target_params['min']) + target_params['min']
                y_pred_original = y_pred * (target_params['max'] - target_params['min']) + target_params['min']
            else:  # zscore
                y_test_original = y_test * target_params['std'] + target_params['mean']
                y_pred_original = y_pred * target_params['std'] + target_params['mean']
            
            # Calculate metrics
            self.evaluator.set_data(y_test_original, y_pred_original, dates=test_df.index.tolist())
            metrics = self.evaluator.calculate_all_metrics()
            
            logger.info(f"Test set performance metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Stage 7: Save model with metadata
            logger.info("\n[7/8] Saving model...")
            
            # Create metadata
            metadata = ModelMetadata(
                version=version,
                model_type=model_type,
                training_date=datetime.now(),
                hyperparameters=best_hyperparams,
                feature_list=feature_list,
                scaling_params=scaling_params,
                performance_metrics=metrics,
                training_data_range=(train_df.index.min(), train_df.index.max()),
                sequence_length=sequence_length
            )
            
            # Save model
            model_path = self.training_pipeline.save_model(model, metadata, version)
            logger.info(f"Model saved to: {model_path}")
            
            # Stage 8: Generate training report
            logger.info("\n[8/8] Generating training report...")
            
            # Generate visualizations
            eval_report = self.evaluator.generate_performance_report(
                report_name=f"training_report_{version}",
                save_json=True,
                save_plots=True
            )
            
            # Generate training history plot if available
            if training_result and training_result.history:
                history_plot = self.visualizer.plot_training_history(
                    history=training_result.history,
                    title=f"{model_type} Training History - {version}",
                    save_path=self.config.REPORTS_DIR / f"training_history_{version}.png"
                )
            
            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Model version: {version}")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Test MAE: {metrics['MAE']:.4f}")
            logger.info(f"Test RMSE: {metrics['RMSE']:.4f}")
            logger.info(f"Test R²: {metrics['R2']:.4f}")
            logger.info("=" * 80)
            
            return model_path
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def run_prediction_pipeline(self, model_version: str, input_data: Optional[pd.DataFrame] = None,
                               horizon: int = 30, output_path: Optional[str] = None,
                               include_confidence: bool = True) -> pd.DataFrame:
        """
        Execute prediction pipeline.
        
        Pipeline stages:
        1. Load trained model
        2. Accept new data or use latest from training
        3. Preprocess input data
        4. Generate predictions
        5. Denormalize predictions
        6. Visualize predictions
        7. Generate prediction report
        
        Args:
            model_version: Version of trained model to load
            input_data: Optional input data (if None, uses latest training data)
            horizon: Number of days to predict (default: 30)
            output_path: Optional path to save predictions CSV
            include_confidence: Whether to include confidence intervals
        
        Returns:
            DataFrame with predictions and timestamps
        
        Requirements: 6.1-6.6, 9.1-9.5
        """
        logger.info("=" * 80)
        logger.info("STARTING PREDICTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Model version: {model_version}")
        logger.info(f"Forecast horizon: {horizon} days")
        
        try:
            # Stage 1: Load trained model
            logger.info("\n[1/7] Loading trained model...")
            model, metadata = self.prediction_service.load_model_by_version(model_version)
            
            model_type = metadata.get('model_type')
            feature_list = metadata.get('feature_list', [])
            sequence_length = metadata.get('sequence_length')
            
            logger.info(f"Loaded model: {model_type}")
            logger.info(f"Features: {len(feature_list)}")
            if sequence_length:
                logger.info(f"Sequence length: {sequence_length}")
            
            # Stage 2: Prepare input data
            logger.info("\n[2/7] Preparing input data...")
            
            if input_data is None:
                logger.info("No input data provided - using placeholder for demo")
                # Create placeholder input features
                n_features = len(feature_list)
                
                if model_type in ['LSTM', 'GRU']:
                    # Create sequence-shaped input
                    input_features = np.random.randn(sequence_length, n_features) * 0.1 + 0.5
                else:
                    # Create flat input
                    input_features = np.random.randn(n_features) * 0.1 + 0.5
            else:
                # Process provided input data
                logger.info(f"Processing input data: {input_data.shape}")
                # Extract features in same order as training
                input_features = input_data[feature_list].values
                
                if model_type in ['LSTM', 'GRU']:
                    # Use last sequence_length rows as input
                    input_features = input_features[-sequence_length:]
            
            # Stage 3: Generate predictions
            logger.info("\n[3/7] Generating predictions...")
            
            start_date = datetime.now()
            predictions_df = self.prediction_service.predict_with_timestamps(
                input_features=input_features,
                start_date=start_date,
                horizon=horizon,
                include_confidence=include_confidence
            )
            
            logger.info(f"Generated {len(predictions_df)} predictions")
            logger.info(f"Date range: {predictions_df['Date'].min()} to {predictions_df['Date'].max()}")
            logger.info(f"Prediction range: [{predictions_df['Prediction'].min():.2f}, {predictions_df['Prediction'].max():.2f}]")
            
            # Stage 4: Visualize predictions
            logger.info("\n[4/7] Generating visualizations...")
            
            # Create time series plot with predictions
            if include_confidence:
                confidence_intervals = predictions_df[['Lower_CI', 'Upper_CI']].values
            else:
                confidence_intervals = None
            
            pred_plot = self.visualizer.plot_time_series_with_predictions(
                actual=np.array([]),  # No actual values for future predictions
                predictions=predictions_df['Prediction'].values,
                confidence_intervals=confidence_intervals,
                dates=predictions_df['Date'].tolist(),
                title=f"Gold Price Forecast - Next {horizon} Days",
                save_path=self.config.REPORTS_DIR / f"prediction_{model_version}_{horizon}d.png"
            )
            
            # Stage 5: Generate prediction report
            logger.info("\n[5/7] Generating prediction report...")
            
            report = self.visualizer.create_prediction_report(
                predictions=predictions_df,
                metrics={'model_version': model_version, 'forecast_horizon': horizon},
                plots=[pred_plot] if pred_plot else None,
                report_name=f"prediction_report_{model_version}",
                save_html=True,
                save_json=True
            )
            
            # Stage 6: Save predictions to CSV
            if output_path:
                logger.info(f"\n[6/7] Saving predictions to {output_path}...")
                predictions_df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")
            else:
                logger.info("\n[6/7] Skipping CSV output (no path provided)")
            
            logger.info("=" * 80)
            logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Generated {len(predictions_df)} predictions")
            logger.info(f"Mean prediction: {predictions_df['Prediction'].mean():.2f}")
            logger.info(f"Std deviation: {predictions_df['Prediction'].std():.2f}")
            logger.info("=" * 80)
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def run_evaluation_pipeline(self, model_version: str, test_data_path: str,
                               compare_version: Optional[str] = None) -> Dict:
        """
        Execute evaluation pipeline.
        
        Pipeline stages:
        1. Load model and test data
        2. Generate predictions on test data
        3. Calculate evaluation metrics
        4. Generate visualizations
        5. Compare with other model versions (optional)
        6. Generate evaluation report
        
        Args:
            model_version: Version of trained model to evaluate
            test_data_path: Path to test data CSV
            compare_version: Optional other model version to compare against
        
        Returns:
            Dictionary with evaluation results
        
        Requirements: 7.1-7.7, 9.1-9.5
        """
        logger.info("=" * 80)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Model version: {model_version}")
        logger.info(f"Test data: {test_data_path}")
        
        try:
            # Stage 1: Load model
            logger.info("\n[1/6] Loading model...")
            model, metadata = self.prediction_service.load_model_by_version(model_version)
            
            model_type = metadata.get('model_type')
            feature_list = metadata.get('feature_list', [])
            scaling_params = metadata.get('scaling_params', {})
            sequence_length = metadata.get('sequence_length')
            
            # Load test data
            logger.info("Loading test data...")
            df_test = self.data_ingestion.load_csv(test_data_path)
            
            # Preprocess test data (same steps as training)
            df_test = self.preprocessor.handle_missing_values(df_test)
            df_test_features = self.feature_engineer.build_feature_set(df_test, handle_nan=True)
            df_test_normalized, _ = self.preprocessor.normalize_features(
                df_test_features,
                method=scaling_params.get('method', 'minmax')
            )
            
            # Prepare features and targets
            if model_type in ['LSTM', 'GRU']:
                X_test, y_test = self.dataset_splitter.prepare_feature_target_split(df_test_normalized)
                X_test, y_test = self.dataset_splitter.create_sequences(X_test, y_test)
            else:
                X_test, y_test = self.dataset_splitter.prepare_feature_target_split(df_test_normalized)
            
            logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
            
            # Stage 2: Generate predictions
            logger.info("\n[2/6] Generating predictions on test data...")
            
            if model_type in ['LSTM', 'GRU']:
                y_pred = model.predict(X_test, verbose=0).flatten()
            else:
                X_test_flat = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) == 3 else X_test
                y_pred = model.predict(X_test_flat)
            
            # Denormalize
            target_params = scaling_params['params'][self.config.TARGET_COLUMN]
            if scaling_params['method'] == 'minmax':
                y_test_original = y_test * (target_params['max'] - target_params['min']) + target_params['min']
                y_pred_original = y_pred * (target_params['max'] - target_params['min']) + target_params['min']
            else:
                y_test_original = y_test * target_params['std'] + target_params['mean']
                y_pred_original = y_pred * target_params['std'] + target_params['mean']
            
            # Stage 3: Calculate metrics
            logger.info("\n[3/6] Calculating evaluation metrics...")
            
            self.evaluator.set_data(y_test_original, y_pred_original)
            metrics = self.evaluator.calculate_all_metrics()
            
            logger.info("Evaluation metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Stage 4: Generate visualizations
            logger.info("\n[4/6] Generating visualizations...")
            
            eval_report = self.evaluator.generate_performance_report(
                report_name=f"evaluation_{model_version}",
                save_json=True,
                save_plots=True
            )
            
            # Stage 5: Compare with other version if provided
            comparison_result = None
            if compare_version:
                logger.info(f"\n[5/6] Comparing with model version {compare_version}...")
                
                comparison_result = self.model_registry.compare_models(model_version, compare_version)
                logger.info(f"Comparison result: {comparison_result}")
            else:
                logger.info("\n[5/6] Skipping model comparison (no compare_version provided)")
            
            # Stage 6: Final report
            logger.info("\n[6/6] Finalizing evaluation report...")
            
            result = {
                'model_version': model_version,
                'model_type': model_type,
                'metrics': metrics,
                'test_samples': len(y_test_original),
                'report_path': eval_report.get('json_path'),
                'plots': {
                    'residuals': eval_report.get('residual_plot_path'),
                    'predictions': eval_report.get('predictions_plot_path')
                },
                'comparison': comparison_result
            }
            
            logger.info("=" * 80)
            logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Test samples: {result['test_samples']}")
            logger.info(f"MAE: {metrics['MAE']:.4f}")
            logger.info(f"RMSE: {metrics['RMSE']:.4f}")
            logger.info(f"R²: {metrics['R2']:.4f}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {str(e)}", exc_info=True)
            raise


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Gold Price Prediction System - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Pipeline mode')
    
    # Training mode parser
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', required=True, help='Path to training data CSV')
    train_parser.add_argument('--model-type', required=True, choices=['LSTM', 'GRU', 'XGBoost', 'RandomForest'],
                             help='Type of model to train')
    train_parser.add_argument('--version', required=True, help='Model version string (e.g., v1.0)')
    train_parser.add_argument('--indicators', action='store_true', help='Include economic indicators')
    train_parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    # Prediction mode parser
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--model-version', required=True, help='Model version to use')
    predict_parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days (default: 30)')
    predict_parser.add_argument('--output', help='Output path for predictions CSV')
    predict_parser.add_argument('--no-confidence', action='store_true', help='Exclude confidence intervals')
    
    # Evaluation mode parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model-version', required=True, help='Model version to evaluate')
    eval_parser.add_argument('--test-data', required=True, help='Path to test data CSV')
    eval_parser.add_argument('--compare', help='Compare with another model version')
    
    return parser.parse_args()


def main():
    """Main entry point for the pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    if args.mode is None:
        print("Error: Please specify a mode (train, predict, or evaluate)")
        print("Use --help for usage information")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = GoldPricePipeline()
    
    try:
        if args.mode == 'train':
            # Training mode
            logger.info(f"Starting training pipeline in mode: {args.mode}")
            
            # Prepare indicators config if requested
            indicators_config = Config.INDICATORS if args.indicators else None
            
            # Run training pipeline
            model_path = pipeline.run_training_pipeline(
                data_path=args.data,
                model_type=args.model_type,
                version=args.version,
                indicators_config=indicators_config,
                hyperparams=None,  # Use defaults
                tune_hyperparams=args.tune
            )
            
            print(f"\nTraining completed successfully!")
            print(f"Model saved to: {model_path}")
            
        elif args.mode == 'predict':
            # Prediction mode
            logger.info(f"Starting prediction pipeline in mode: {args.mode}")
            
            # Run prediction pipeline
            predictions_df = pipeline.run_prediction_pipeline(
                model_version=args.model_version,
                input_data=None,  # Use placeholder
                horizon=args.horizon,
                output_path=args.output,
                include_confidence=not args.no_confidence
            )
            
            print(f"\nPrediction completed successfully!")
            print(f"Generated {len(predictions_df)} predictions")
            print(f"\nFirst few predictions:")
            print(predictions_df.head(10))
            
            if args.output:
                print(f"\nPredictions saved to: {args.output}")
            
        elif args.mode == 'evaluate':
            # Evaluation mode
            logger.info(f"Starting evaluation pipeline in mode: {args.mode}")
            
            # Run evaluation pipeline
            result = pipeline.run_evaluation_pipeline(
                model_version=args.model_version,
                test_data_path=args.test_data,
                compare_version=args.compare
            )
            
            print(f"\nEvaluation completed successfully!")
            print(f"\nPerformance Metrics:")
            for metric, value in result['metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nReport saved to: {result['report_path']}")
            
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        print("Check logs for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
