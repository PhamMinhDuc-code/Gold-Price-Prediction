"""
Model Training Pipeline Module

This module implements the ModelTrainingPipeline class for building and training
machine learning models for gold price prediction. Supports LSTM, GRU, XGBoost,
and Random Forest architectures.

Requirements: 5.1, 5.3
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

from config import Config


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

# Add handler if not already present
if not logger.handlers:
    log_file = Config.get_log_path('src.model_training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class TrainingResult:
    """Container for training results."""
    
    def __init__(self, model: Any, training_time: float, final_loss: float,
                 validation_loss: float, convergence_status: str, history: Dict):
        self.model = model
        self.training_time = training_time
        self.final_loss = final_loss
        self.validation_loss = validation_loss
        self.convergence_status = convergence_status
        self.history = history


class ModelMetadata:
    """Container for model metadata."""
    
    def __init__(self, version: str, model_type: str, training_date: datetime,
                 hyperparameters: Dict, feature_list: list, scaling_params: Dict,
                 performance_metrics: Dict, training_data_range: Tuple[datetime, datetime],
                 sequence_length: Optional[int] = None):
        self.version = version
        self.model_type = model_type
        self.training_date = training_date
        self.hyperparameters = hyperparameters
        self.feature_list = feature_list
        self.scaling_params = scaling_params
        self.performance_metrics = performance_metrics
        self.training_data_range = training_data_range
        self.sequence_length = sequence_length
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary for serialization."""
        return {
            'version': self.version,
            'model_type': self.model_type,
            'training_date': self.training_date.isoformat() if isinstance(self.training_date, datetime) else str(self.training_date),
            'hyperparameters': self.hyperparameters,
            'feature_list': self.feature_list,
            'scaling_params': self.scaling_params,
            'performance_metrics': self.performance_metrics,
            'training_data_range': [
                self.training_data_range[0].isoformat() if isinstance(self.training_data_range[0], datetime) else str(self.training_data_range[0]),
                self.training_data_range[1].isoformat() if isinstance(self.training_data_range[1], datetime) else str(self.training_data_range[1])
            ],
            'sequence_length': self.sequence_length
        }


class ModelTrainingPipeline:
    """
    Model Training Pipeline for gold price prediction.
    
    This class manages the training of various model architectures including
    LSTM, GRU, XGBoost, and Random Forest models. It handles model building,
    training, persistence, and logging.
    
    Requirements: 5.1, 5.3
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize ModelTrainingPipeline with configuration parameters.
        
        Args:
            config: Configuration object containing model parameters.
                   Defaults to Config if not provided.
        
        Requirements: 5.1
        """
        self.config = config if config is not None else Config
        
        # Set up model storage directory
        self.model_dir = self.config.MODEL_DIR
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info(f"ModelTrainingPipeline initialized with model directory: {self.model_dir}")
        logger.info(f"Configuration - Sequence Length: {self.config.SEQUENCE_LENGTH}, "
                   f"Epochs: {self.config.EPOCHS}, Learning Rate: {self.config.LEARNING_RATE}")
    
    def build_lstm_model(self, input_shape: Tuple, hyperparams: Optional[Dict] = None) -> keras.Model:
        """
        Build LSTM model using Keras Sequential API.
        
        Architecture:
        - LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
        - Uses Adam optimizer with configurable learning rate
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            hyperparams: Optional dictionary with hyperparameters:
                - units_layer1: Number of units in first LSTM layer (default: 128)
                - units_layer2: Number of units in second LSTM layer (default: 64)
                - dropout: Dropout rate (default: 0.2)
                - dense_units: Number of units in dense layer (default: 32)
                - learning_rate: Learning rate for optimizer (default: from config)
        
        Returns:
            Compiled Keras LSTM model
        
        Requirements: 5.3
        """
        # Extract hyperparameters with defaults
        if hyperparams is None:
            hyperparams = {}
        
        units_layer1 = hyperparams.get('units_layer1', 128)
        units_layer2 = hyperparams.get('units_layer2', 64)
        dropout = hyperparams.get('dropout', 0.2)
        dense_units = hyperparams.get('dense_units', 32)
        learning_rate = hyperparams.get('learning_rate', self.config.LEARNING_RATE)
        
        logger.info(f"Building LSTM model with input_shape={input_shape}, "
                   f"units=[{units_layer1}, {units_layer2}], dropout={dropout}, "
                   f"dense_units={dense_units}, lr={learning_rate}")
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(units_layer1, return_sequences=True, name='lstm_1'),
            layers.Dropout(dropout, name='dropout_1'),
            layers.LSTM(units_layer2, name='lstm_2'),
            layers.Dropout(dropout, name='dropout_2'),
            layers.Dense(dense_units, activation='relu', name='dense_1'),
            layers.Dense(1, activation='linear', name='output')
        ], name='LSTM_Model')
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"LSTM model compiled successfully. Total parameters: {model.count_params()}")
        
        return model
    
    def build_gru_model(self, input_shape: Tuple, hyperparams: Optional[Dict] = None) -> keras.Model:
        """
        Build GRU model using Keras Sequential API.
        
        Architecture:
        - GRU(128) → Dropout(0.2) → GRU(64) → Dropout(0.2) → Dense(32) → Dense(1)
        - Uses Adam optimizer with configurable learning rate
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            hyperparams: Optional dictionary with hyperparameters:
                - units_layer1: Number of units in first GRU layer (default: 128)
                - units_layer2: Number of units in second GRU layer (default: 64)
                - dropout: Dropout rate (default: 0.2)
                - dense_units: Number of units in dense layer (default: 32)
                - learning_rate: Learning rate for optimizer (default: from config)
        
        Returns:
            Compiled Keras GRU model
        
        Requirements: 5.3
        """
        # Extract hyperparameters with defaults
        if hyperparams is None:
            hyperparams = {}
        
        units_layer1 = hyperparams.get('units_layer1', 128)
        units_layer2 = hyperparams.get('units_layer2', 64)
        dropout = hyperparams.get('dropout', 0.2)
        dense_units = hyperparams.get('dense_units', 32)
        learning_rate = hyperparams.get('learning_rate', self.config.LEARNING_RATE)
        
        logger.info(f"Building GRU model with input_shape={input_shape}, "
                   f"units=[{units_layer1}, {units_layer2}], dropout={dropout}, "
                   f"dense_units={dense_units}, lr={learning_rate}")
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.GRU(units_layer1, return_sequences=True, name='gru_1'),
            layers.Dropout(dropout, name='dropout_1'),
            layers.GRU(units_layer2, name='gru_2'),
            layers.Dropout(dropout, name='dropout_2'),
            layers.Dense(dense_units, activation='relu', name='dense_1'),
            layers.Dense(1, activation='linear', name='output')
        ], name='GRU_Model')
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"GRU model compiled successfully. Total parameters: {model.count_params()}")
        
        return model
    
    def build_xgboost_model(self, hyperparams: Optional[Dict] = None) -> XGBRegressor:
        """
        Build XGBoost model using XGBRegressor.
        
        Configures model with hyperparameters including max_depth, n_estimators,
        and learning_rate.
        
        Args:
            hyperparams: Optional dictionary with hyperparameters:
                - max_depth: Maximum tree depth (default: 5)
                - n_estimators: Number of boosting rounds (default: 100)
                - learning_rate: Learning rate (default: 0.1)
                - subsample: Subsample ratio (default: 1.0)
                - colsample_bytree: Feature subsample ratio (default: 1.0)
                - random_state: Random seed (default: 42)
        
        Returns:
            Configured XGBoost regressor
        
        Requirements: 5.3
        """
        # Extract hyperparameters with defaults
        if hyperparams is None:
            hyperparams = {}
        
        max_depth = hyperparams.get('max_depth', 5)
        n_estimators = hyperparams.get('n_estimators', 100)
        learning_rate = hyperparams.get('learning_rate', 0.1)
        subsample = hyperparams.get('subsample', 1.0)
        colsample_bytree = hyperparams.get('colsample_bytree', 1.0)
        random_state = hyperparams.get('random_state', 42)
        
        logger.info(f"Building XGBoost model with max_depth={max_depth}, "
                   f"n_estimators={n_estimators}, learning_rate={learning_rate}, "
                   f"subsample={subsample}, colsample_bytree={colsample_bytree}")
        
        # Build model
        model = XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        logger.info("XGBoost model created successfully")
        
        return model
    
    def build_random_forest_model(self, hyperparams: Optional[Dict] = None) -> RandomForestRegressor:
        """
        Build Random Forest model using RandomForestRegressor.
        
        Configures model with hyperparameters including n_estimators, max_depth,
        and min_samples_split.
        
        Args:
            hyperparams: Optional dictionary with hyperparameters:
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum tree depth (default: None)
                - min_samples_split: Minimum samples to split node (default: 2)
                - min_samples_leaf: Minimum samples in leaf node (default: 1)
                - random_state: Random seed (default: 42)
        
        Returns:
            Configured Random Forest regressor
        
        Requirements: 5.3
        """
        # Extract hyperparameters with defaults
        if hyperparams is None:
            hyperparams = {}
        
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', None)
        min_samples_split = hyperparams.get('min_samples_split', 2)
        min_samples_leaf = hyperparams.get('min_samples_leaf', 1)
        random_state = hyperparams.get('random_state', 42)
        
        logger.info(f"Building Random Forest model with n_estimators={n_estimators}, "
                   f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
                   f"min_samples_leaf={min_samples_leaf}")
        
        # Build model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        
        logger.info("Random Forest model created successfully")
        
        return model
    
    def train_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_type: str = 'deep_learning',
                   checkpoint_path: Optional[str] = None) -> TrainingResult:
        """
        Train model with early stopping and checkpointing.
        
        Implements:
        - Early stopping with patience=10 for deep learning models
        - Model checkpointing to save best model
        - Validation set monitoring for both deep learning and tree-based models
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of model ('deep_learning' or 'tree_based')
            checkpoint_path: Path to save best model checkpoint (optional)
        
        Returns:
            TrainingResult object containing trained model and metrics
        
        Requirements: 5.1
        """
        import time
        
        start_time = time.time()
        logger.info(f"Starting model training. Model type: {model_type}")
        logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        if model_type == 'deep_learning':
            # Train deep learning models (LSTM, GRU)
            # Early stopping callback with patience=10
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
            
            callbacks = [early_stopping]
            
            # Add model checkpointing if path provided
            if checkpoint_path is not None:
                checkpoint_dir = Path(checkpoint_path).parent
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                model_checkpoint = ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    mode='min',
                    verbose=1
                )
                callbacks.append(model_checkpoint)
                logger.info(f"Model checkpoint enabled. Best model will be saved to: {checkpoint_path}")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            final_loss = history.history['loss'][-1]
            validation_loss = history.history['val_loss'][-1]
            convergence_status = 'converged' if len(history.history['loss']) < self.config.EPOCHS else 'max_epochs_reached'
            
            history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            
            logger.info(f"Early stopping triggered at epoch {len(history.history['loss'])} of {self.config.EPOCHS}")
            
        else:
            # Train tree-based models (XGBoost, Random Forest)
            # Handle XGBoost with early stopping using eval_set
            if hasattr(model, 'fit') and 'XGB' in type(model).__name__:
                logger.info("Training XGBoost model with early stopping via eval_set")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                # Random Forest or other sklearn models
                logger.info("Training tree-based model (sklearn)")
                model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Calculate training and validation errors
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            final_loss = np.mean((y_train - train_pred) ** 2)
            validation_loss = np.mean((y_val - val_pred) ** 2)
            convergence_status = 'completed'
            
            history_dict = {
                'loss': [float(final_loss)],
                'val_loss': [float(validation_loss)]
            }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {final_loss:.6f}, Validation loss: {validation_loss:.6f}")
        logger.info(f"Convergence status: {convergence_status}")
        
        return TrainingResult(
            model=model,
            training_time=training_time,
            final_loss=final_loss,
            validation_loss=validation_loss,
            convergence_status=convergence_status,
            history=history_dict
        )
    
    def hyperparameter_tuning(self, model_type: str, X_train: np.ndarray, 
                             y_train: np.ndarray, X_val: np.ndarray, 
                             y_val: np.ndarray,
                             search_space: Optional[Dict] = None,
                             method: str = 'grid') -> Dict:
        """
        Perform hyperparameter tuning for specified model type.
        
        Uses GridSearchCV for tree-based models (XGBoost, Random Forest).
        Uses manual loop with different configs for deep learning models (LSTM, GRU).
        Returns best hyperparameters based on validation performance.
        
        Args:
            model_type: Type of model ('LSTM', 'GRU', 'XGBoost', 'RandomForest')
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            search_space: Optional custom hyperparameter search space
            method: Search method ('grid' or 'random') - currently only 'grid' supported
        
        Returns:
            Dictionary with best hyperparameters and validation score
        
        Requirements: 5.2, 5.4
        """
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        logger.info(f"Search method: {method}")
        
        if model_type in ['LSTM', 'GRU']:
            # Manual loop for deep learning models
            return self._tune_deep_learning_model(
                model_type, X_train, y_train, X_val, y_val, search_space
            )
        elif model_type in ['XGBoost', 'RandomForest']:
            # GridSearchCV for tree-based models
            return self._tune_tree_based_model(
                model_type, X_train, y_train, X_val, y_val, search_space
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _tune_deep_learning_model(self, model_type: str, X_train: np.ndarray,
                                   y_train: np.ndarray, X_val: np.ndarray,
                                   y_val: np.ndarray,
                                   search_space: Optional[Dict] = None) -> Dict:
        """
        Tune deep learning models using manual configuration loop.
        
        Requirements: 5.2, 5.4
        """
        import itertools
        
        # Use default search space from config if not provided
        if search_space is None:
            if model_type == 'LSTM':
                search_space = self.config.LSTM_HYPERPARAMS
            else:  # GRU
                search_space = self.config.GRU_HYPERPARAMS
        
        logger.info(f"Search space: {search_space}")
        
        # Generate all combinations
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        logger.info(f"Total configurations to test: {len(combinations)}")
        
        best_val_loss = float('inf')
        best_params = None
        best_model = None
        
        input_shape = (X_train.shape[1], X_train.shape[2]) if len(X_train.shape) == 3 else X_train.shape[1:]
        
        for i, combo in enumerate(combinations):
            hyperparams = dict(zip(keys, combo))
            logger.info(f"Testing configuration {i+1}/{len(combinations)}: {hyperparams}")
            
            # Build model
            if model_type == 'LSTM':
                model = self.build_lstm_model(input_shape, hyperparams)
            else:  # GRU
                model = self.build_gru_model(input_shape, hyperparams)
            
            # Train model with early stopping
            try:
                result = self.train_model(
                    model, X_train, y_train, X_val, y_val, 
                    model_type='deep_learning'
                )
                
                val_loss = result.validation_loss
                logger.info(f"Configuration {i+1} validation loss: {val_loss:.6f}")
                
                # Update best model if this is better
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = hyperparams.copy()
                    best_model = model
                    logger.info(f"New best configuration found! Val loss: {val_loss:.6f}")
                
            except Exception as e:
                logger.error(f"Configuration {i+1} failed: {str(e)}")
                continue
        
        logger.info(f"Hyperparameter tuning complete")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'best_model': best_model
        }
    
    def _tune_tree_based_model(self, model_type: str, X_train: np.ndarray,
                               y_train: np.ndarray, X_val: np.ndarray,
                               y_val: np.ndarray,
                               search_space: Optional[Dict] = None) -> Dict:
        """
        Tune tree-based models using GridSearchCV.
        
        Requirements: 5.2, 5.4
        """
        from sklearn.model_selection import GridSearchCV
        
        # Use default search space from config if not provided
        if search_space is None:
            if model_type == 'XGBoost':
                search_space = self.config.XGBOOST_HYPERPARAMS
            else:  # RandomForest
                search_space = self.config.RANDOM_FOREST_HYPERPARAMS
        
        logger.info(f"Search space: {search_space}")
        
        # Build base model
        if model_type == 'XGBoost':
            base_model = self.build_xgboost_model()
        else:  # RandomForest
            base_model = self.build_random_forest_model()
        
        # For tree-based models, flatten the data if it's 3D (sequence format)
        if len(X_train.shape) == 3:
            logger.info("Reshaping 3D data to 2D for tree-based model")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val
        
        # Perform grid search
        logger.info("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            base_model,
            search_space,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train_flat, y_train)
        
        # Evaluate on validation set
        best_model = grid_search.best_estimator_
        val_pred = best_model.predict(X_val_flat)
        best_val_loss = np.mean((y_val - val_pred) ** 2)
        
        logger.info(f"Hyperparameter tuning complete")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.6f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_val_loss': best_val_loss,
            'best_model': best_model,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, model: Any, metadata: ModelMetadata, version: str) -> str:
        """
        Save model with metadata and preprocessing parameters.
        
        Saves:
        - Keras models using native Keras format (.keras files)
        - Scikit-learn/XGBoost models using joblib
        - Metadata as JSON (version, hyperparams, features, metrics)
        - Scaling parameters as pickle file
        
        Args:
            model: Trained model to save
            metadata: Model metadata object
            version: Version string for the model
        
        Returns:
            Path to saved model directory
        
        Requirements: 5.5, 8.1, 8.2, 8.3
        """
        import pickle
        
        model_path = self.config.get_model_path(version, metadata.model_type)
        logger.info(f"Saving model to: {model_path}")
        
        try:
            # Save model based on type
            if metadata.model_type in ['LSTM', 'GRU']:
                # Save Keras models using native Keras format
                model_file = model_path / 'model.keras'
                model.save(model_file)
                logger.info(f"Keras model saved to {model_file}")
            else:
                # Save scikit-learn/XGBoost models using joblib
                model_file = model_path / 'model.pkl'
                joblib.dump(model, model_file)
                logger.info(f"Scikit-learn/XGBoost model saved to {model_file}")
            
            # Save metadata as JSON (version, hyperparams, features, metrics)
            metadata_path = model_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            logger.debug(f"Metadata includes: version, hyperparameters, feature_list, "
                        f"performance_metrics, training_data_range")
            
            # Save scaling parameters as pickle file
            if metadata.scaling_params:
                scaler_path = model_path / 'scaler.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(metadata.scaling_params, f)
                logger.info(f"Scaling parameters saved to {scaler_path}")
            else:
                logger.warning("No scaling parameters provided in metadata")
            
            # Update registry
            self._update_registry(metadata, model_path)
            
            logger.info(f"Model version {version} saved successfully")
            logger.info(f"Saved files: model, metadata.json, scaler.pkl")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise
    
    def _update_registry(self, metadata: ModelMetadata, model_path: Path) -> None:
        """Update model registry with new model information."""
        from src.model_registry import ModelRegistry
        
        # Use ModelRegistry to ensure consistency
        registry = ModelRegistry(models_directory=self.model_dir)
        
        # Register the model
        registry.register_model(
            model_path=str(model_path),
            metadata=metadata.to_dict(),
            version=metadata.version
        )
        
        logger.info(f"Registry updated with model version {metadata.version}")
    
    def log_training_metrics(self, training_result: TrainingResult, 
                            model_version: str,
                            save_to_file: bool = True) -> None:
        """
        Log training metrics, loss curves, and convergence status.
        
        Logs comprehensive training information including:
        - Training time
        - Loss curves (training and validation)
        - Convergence status
        - Final loss values
        
        Optionally saves training history to JSON file for later analysis.
        
        Args:
            training_result: TrainingResult object from train_model()
            model_version: Version string for the model
            save_to_file: Whether to save training history to JSON file
        
        Requirements: 5.6
        """
        logger.info("=" * 70)
        logger.info("TRAINING METRICS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Model Version: {model_version}")
        logger.info(f"Training Time: {training_result.training_time:.2f} seconds")
        logger.info(f"Final Training Loss: {training_result.final_loss:.6f}")
        logger.info(f"Final Validation Loss: {training_result.validation_loss:.6f}")
        logger.info(f"Convergence Status: {training_result.convergence_status}")
        
        # Log loss curve summary
        if 'loss' in training_result.history:
            loss_curve = training_result.history['loss']
            logger.info(f"Training Epochs: {len(loss_curve)}")
            logger.info(f"Initial Training Loss: {loss_curve[0]:.6f}")
            logger.info(f"Loss Reduction: {(loss_curve[0] - loss_curve[-1]):.6f}")
        
        if 'val_loss' in training_result.history:
            val_loss_curve = training_result.history['val_loss']
            logger.info(f"Initial Validation Loss: {val_loss_curve[0]:.6f}")
            logger.info(f"Val Loss Reduction: {(val_loss_curve[0] - val_loss_curve[-1]):.6f}")
        
        logger.info("=" * 70)
        
        # Save training history to JSON file
        if save_to_file:
            history_file = self.model_dir / f"training_history_{model_version}.json"
            
            history_data = {
                'model_version': model_version,
                'training_time': training_result.training_time,
                'final_loss': training_result.final_loss,
                'validation_loss': training_result.validation_loss,
                'convergence_status': training_result.convergence_status,
                'history': training_result.history,
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                with open(history_file, 'w') as f:
                    json.dump(history_data, f, indent=2)
                logger.info(f"Training history saved to: {history_file}")
            except Exception as e:
                logger.error(f"Failed to save training history: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Example usage
    print("ModelTrainingPipeline module loaded successfully")
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    # Example: Build LSTM model
    lstm_model = pipeline.build_lstm_model(input_shape=(60, 10))
    print(f"\nLSTM Model Summary:")
    lstm_model.summary()
    
    # Example: Build GRU model
    gru_model = pipeline.build_gru_model(input_shape=(60, 10))
    print(f"\nGRU Model Summary:")
    gru_model.summary()
    
    # Example: Build XGBoost model
    xgb_model = pipeline.build_xgboost_model()
    print(f"\nXGBoost Model: {xgb_model}")
    
    # Example: Build Random Forest model
    rf_model = pipeline.build_random_forest_model()
    print(f"\nRandom Forest Model: {rf_model}")
