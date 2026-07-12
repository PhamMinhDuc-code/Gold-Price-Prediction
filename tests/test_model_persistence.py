"""
Unit Tests for Model Persistence and Registry

This module tests the model persistence, loading, and registry functionality.

Tests:
- Model save/load round-trip preserves predictions
- Metadata is correctly saved and loaded
- Model registry tracks versions correctly
- Compatibility validation works

Requirements: 8.1-8.5
"""

import pytest
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from src.model_training import ModelTrainingPipeline, ModelMetadata
from src.model_registry import ModelRegistry, ModelInfo


class TestModelPersistence:
    """Test model save/load functionality."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample model metadata."""
        return ModelMetadata(
            version='v1.0.0_test',
            model_type='LSTM',
            training_date=datetime.now(),
            hyperparameters={'units_layer1': 128, 'dropout': 0.2},
            feature_list=['Close', 'Volume', 'Close_lag_1'],
            scaling_params={'min': 0.0, 'max': 2000.0},
            performance_metrics={'mae': 10.5, 'rmse': 15.2, 'r2': 0.85},
            training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31)),
            sequence_length=60
        )
    
    @pytest.fixture
    def sample_lstm_model(self):
        """Create a small LSTM model for testing."""
        from src.model_training import ModelTrainingPipeline
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_lstm_model(
            input_shape=(60, 3),
            hyperparams={'units_layer1': 32, 'units_layer2': 16, 'dropout': 0.2}
        )
        return model
    
    @pytest.fixture
    def sample_xgboost_model(self):
        """Create a small XGBoost model for testing."""
        from src.model_training import ModelTrainingPipeline
        pipeline = ModelTrainingPipeline()
        model = pipeline.build_xgboost_model(
            hyperparams={'max_depth': 3, 'n_estimators': 10}
        )
        # Fit with dummy data
        X_dummy = np.random.randn(50, 5)
        y_dummy = np.random.randn(50)
        model.fit(X_dummy, y_dummy)
        return model
    
    def test_lstm_save_load_roundtrip(self, temp_model_dir, sample_lstm_model, sample_metadata):
        """
        Test that LSTM model save/load preserves predictions.
        
        Requirements: 8.1-8.4
        """
        # Override config for testing
        from config import Config
        original_model_dir = Config.MODEL_DIR
        Config.MODEL_DIR = temp_model_dir
        
        try:
            # Initialize pipeline with temp directory
            pipeline = ModelTrainingPipeline(config=Config)
            
            # Generate sample input
            X_test = np.random.randn(10, 60, 3)
            
            # Get predictions before saving
            predictions_before = sample_lstm_model.predict(X_test, verbose=0)
            
            # Save model
            model_path = pipeline.save_model(
                model=sample_lstm_model,
                metadata=sample_metadata,
                version='v1.0.0_test'
            )
            
            # Verify files exist
            model_dir = Path(model_path)
            assert ((model_dir / 'model.keras').exists() or (model_dir / 'model.h5').exists()), "Model file not created"
            assert (model_dir / 'metadata.json').exists(), "Metadata file not created"
            assert (model_dir / 'scaler.pkl').exists(), "Scaler file not created"
            
            # Load model using registry
            registry = ModelRegistry(models_directory=temp_model_dir)
            loaded_model, loaded_metadata = registry.load_model_by_version('v1.0.0_test')
            
            # Get predictions after loading
            predictions_after = loaded_model.predict(X_test, verbose=0)
            
            # Verify predictions are identical
            np.testing.assert_allclose(
                predictions_before,
                predictions_after,
                rtol=1e-5,
                err_msg="Predictions differ after save/load"
            )
            
        finally:
            # Restore original config
            Config.MODEL_DIR = original_model_dir
    
    def test_xgboost_save_load_roundtrip(self, temp_model_dir, sample_xgboost_model):
        """
        Test that XGBoost model save/load preserves predictions.
        
        Requirements: 8.1-8.4
        """
        from config import Config
        original_model_dir = Config.MODEL_DIR
        Config.MODEL_DIR = temp_model_dir
        
        try:
            pipeline = ModelTrainingPipeline(config=Config)
            
            # Create metadata for XGBoost model
            metadata = ModelMetadata(
                version='v1.0.0_xgb_test',
                model_type='XGBoost',
                training_date=datetime.now(),
                hyperparameters={'max_depth': 3, 'n_estimators': 10},
                feature_list=['f1', 'f2', 'f3', 'f4', 'f5'],
                scaling_params={'mean': 0.0, 'std': 1.0},
                performance_metrics={'mae': 5.2, 'rmse': 7.1},
                training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31))
            )
            
            # Generate sample input
            X_test = np.random.randn(10, 5)
            
            # Get predictions before saving
            predictions_before = sample_xgboost_model.predict(X_test)
            
            # Save model
            model_path = pipeline.save_model(
                model=sample_xgboost_model,
                metadata=metadata,
                version='v1.0.0_xgb_test'
            )
            
            # Verify files exist
            model_dir = Path(model_path)
            assert (model_dir / 'model.pkl').exists(), "Model file not created"
            assert (model_dir / 'metadata.json').exists(), "Metadata file not created"
            
            # Load model
            registry = ModelRegistry(models_directory=temp_model_dir)
            loaded_model, loaded_metadata = registry.load_model_by_version('v1.0.0_xgb_test')
            
            # Get predictions after loading
            predictions_after = loaded_model.predict(X_test)
            
            # Verify predictions are identical
            np.testing.assert_allclose(
                predictions_before,
                predictions_after,
                rtol=1e-5,
                err_msg="Predictions differ after save/load"
            )
            
        finally:
            Config.MODEL_DIR = original_model_dir
    
    def test_metadata_save_load(self, temp_model_dir, sample_lstm_model, sample_metadata):
        """
        Test that metadata is correctly saved and loaded.
        
        Requirements: 8.2, 8.4
        """
        from config import Config
        original_model_dir = Config.MODEL_DIR
        Config.MODEL_DIR = temp_model_dir
        
        try:
            pipeline = ModelTrainingPipeline(config=Config)
            
            # Save model
            pipeline.save_model(
                model=sample_lstm_model,
                metadata=sample_metadata,
                version='v1.0.0_test'
            )
            
            # Load metadata
            registry = ModelRegistry(models_directory=temp_model_dir)
            model_path = temp_model_dir / 'model_v1.0.0_test'
            loaded_metadata = registry.load_metadata(model_path)
            
            # Verify all metadata fields
            assert loaded_metadata['version'] == 'v1.0.0_test'
            assert loaded_metadata['model_type'] == 'LSTM'
            assert loaded_metadata['sequence_length'] == 60
            assert 'units_layer1' in loaded_metadata['hyperparameters']
            assert loaded_metadata['hyperparameters']['units_layer1'] == 128
            assert 'Close' in loaded_metadata['feature_list']
            assert loaded_metadata['performance_metrics']['mae'] == 10.5
            assert loaded_metadata['performance_metrics']['rmse'] == 15.2
            assert loaded_metadata['performance_metrics']['r2'] == 0.85
            
            # Verify scaling params
            scaling_params = registry.load_scaling_params(model_path)
            assert scaling_params['min'] == 0.0
            assert scaling_params['max'] == 2000.0
            
        finally:
            Config.MODEL_DIR = original_model_dir


class TestModelRegistry:
    """Test model registry functionality."""
    
    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for registry."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_registry(self, temp_registry_dir):
        """Create a sample registry with test models."""
        registry = ModelRegistry(models_directory=temp_registry_dir)
        return registry
    
    def test_registry_initialization(self, temp_registry_dir):
        """
        Test that registry.json is created if not exists.
        
        Requirements: 8.1
        """
        registry = ModelRegistry(models_directory=temp_registry_dir)
        
        # Verify registry file exists
        registry_file = temp_registry_dir / 'registry.json'
        assert registry_file.exists(), "Registry file not created"
        
        # Verify structure
        with open(registry_file, 'r') as f:
            data = json.load(f)
        
        assert 'created_at' in data
        assert 'last_updated' in data
        assert 'models' in data
        assert isinstance(data['models'], list)
    
    def test_register_model(self, sample_registry, temp_registry_dir):
        """
        Test that register_model adds model to registry.
        
        Requirements: 8.1
        """
        # Create dummy model directory
        model_path = temp_registry_dir / 'model_v1.0.0'
        model_path.mkdir()
        
        metadata = {
            'version': 'v1.0.0',
            'model_type': 'LSTM',
            'training_date': datetime.now().isoformat(),
            'performance_metrics': {'mae': 10.0, 'rmse': 15.0}
        }
        
        # Register model
        version_id = sample_registry.register_model(
            model_path=str(model_path),
            metadata=metadata,
            version='v1.0.0'
        )
        
        assert version_id == 'v1.0.0'
        
        # Verify model is in registry
        models = sample_registry.list_models()
        assert len(models) == 1
        assert models[0].version == 'v1.0.0'
        assert models[0].model_type == 'LSTM'
    
    def test_list_models(self, sample_registry, temp_registry_dir):
        """
        Test that list_models shows all versions.
        
        Requirements: 8.1
        """
        # Register multiple models
        for i in range(3):
            model_path = temp_registry_dir / f'model_v1.0.{i}'
            model_path.mkdir()
            
            metadata = {
                'version': f'v1.0.{i}',
                'model_type': 'LSTM',
                'training_date': datetime.now().isoformat(),
                'performance_metrics': {'mae': 10.0 + i, 'rmse': 15.0 + i}
            }
            
            sample_registry.register_model(
                model_path=str(model_path),
                metadata=metadata,
                version=f'v1.0.{i}'
            )
        
        # List models
        models = sample_registry.list_models()
        
        assert len(models) == 3
        versions = [m.version for m in models]
        assert 'v1.0.0' in versions
        assert 'v1.0.1' in versions
        assert 'v1.0.2' in versions
    
    def test_load_latest_model(self, sample_registry, temp_registry_dir):
        """
        Test that load_latest_model returns most recent version.
        
        Requirements: 8.4
        """
        from src.model_training import ModelTrainingPipeline
        from config import Config
        
        original_model_dir = Config.MODEL_DIR
        Config.MODEL_DIR = temp_registry_dir
        
        try:
            pipeline = ModelTrainingPipeline(config=Config)
            
            # Create and save multiple models with different dates
            for i in range(3):
                model = pipeline.build_lstm_model(input_shape=(60, 3))
                
                metadata = ModelMetadata(
                    version=f'v1.0.{i}',
                    model_type='LSTM',
                    training_date=datetime(2023, 1, i+1),  # Different dates
                    hyperparameters={'units': 32},
                    feature_list=['f1', 'f2', 'f3'],
                    scaling_params={'min': 0, 'max': 1},
                    performance_metrics={'mae': 10.0},
                    training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31)),
                    sequence_length=60
                )
                
                pipeline.save_model(model, metadata, f'v1.0.{i}')
            
            # Load latest model
            loaded_model, loaded_metadata = sample_registry.load_latest_model()
            
            # Should be v1.0.2 (latest date)
            assert loaded_metadata['version'] == 'v1.0.2'
            
        finally:
            Config.MODEL_DIR = original_model_dir
    
    def test_validate_model_compatibility(self, sample_registry):
        """
        Test that validate_model_compatibility checks schema.
        
        Requirements: 8.5
        """
        # Valid metadata
        valid_metadata = {
            'version': 'v1.0.0',
            'model_type': 'LSTM',
            'feature_list': ['f1', 'f2', 'f3'],
            'scaling_params': {'min': 0, 'max': 1},
            'sequence_length': 60
        }
        
        assert sample_registry.validate_model_compatibility(valid_metadata) == True
        
        # Missing required field
        invalid_metadata_1 = {
            'version': 'v1.0.0',
            'model_type': 'LSTM',
            # Missing feature_list
            'scaling_params': {'min': 0, 'max': 1}
        }
        
        assert sample_registry.validate_model_compatibility(invalid_metadata_1) == False
        
        # Unsupported model type
        invalid_metadata_2 = {
            'version': 'v1.0.0',
            'model_type': 'UnsupportedType',
            'feature_list': ['f1'],
            'scaling_params': {'min': 0, 'max': 1}
        }
        
        assert sample_registry.validate_model_compatibility(invalid_metadata_2) == False
        
        # Empty feature list
        invalid_metadata_3 = {
            'version': 'v1.0.0',
            'model_type': 'LSTM',
            'feature_list': [],
            'scaling_params': {'min': 0, 'max': 1}
        }
        
        assert sample_registry.validate_model_compatibility(invalid_metadata_3) == False
        
        # LSTM without sequence_length
        invalid_metadata_4 = {
            'version': 'v1.0.0',
            'model_type': 'LSTM',
            'feature_list': ['f1'],
            'scaling_params': {'min': 0, 'max': 1}
            # Missing sequence_length
        }
        
        assert sample_registry.validate_model_compatibility(invalid_metadata_4) == False


class TestModelRegistryTracking:
    """Test that registry correctly tracks versions."""
    
    @pytest.fixture
    def temp_tracking_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_registry_tracks_versions_correctly(self, temp_tracking_dir):
        """
        Test that model registry tracks versions correctly.
        
        Requirements: 8.1
        """
        from src.model_training import ModelTrainingPipeline
        from config import Config
        
        original_model_dir = Config.MODEL_DIR
        Config.MODEL_DIR = temp_tracking_dir
        
        try:
            pipeline = ModelTrainingPipeline(config=Config)
            registry = ModelRegistry(models_directory=temp_tracking_dir)
            
            # Save multiple model versions
            versions = ['v1.0.0', 'v1.0.1', 'v1.1.0', 'v2.0.0']
            
            for version in versions:
                model = pipeline.build_lstm_model(input_shape=(60, 3))
                
                metadata = ModelMetadata(
                    version=version,
                    model_type='LSTM',
                    training_date=datetime.now(),
                    hyperparameters={'units': 32},
                    feature_list=['f1', 'f2', 'f3'],
                    scaling_params={'min': 0, 'max': 1},
                    performance_metrics={'mae': 10.0, 'rmse': 15.0, 'r2': 0.8},
                    training_data_range=(datetime(2020, 1, 1), datetime(2022, 12, 31)),
                    sequence_length=60
                )
                
                pipeline.save_model(model, metadata, version)
            
            # List all models
            models = registry.list_models()
            
            # Verify all versions are tracked
            assert len(models) == 4
            
            tracked_versions = [m.version for m in models]
            for version in versions:
                assert version in tracked_versions, f"Version {version} not tracked"
            
            # Verify each model has performance metrics
            for model_info in models:
                assert 'mae' in model_info.performance_metrics
                assert 'rmse' in model_info.performance_metrics
                assert 'r2' in model_info.performance_metrics
            
            # Verify we can load each version
            for version in versions:
                loaded_model, loaded_metadata = registry.load_model_by_version(version)
                assert loaded_metadata['version'] == version
                
        finally:
            Config.MODEL_DIR = original_model_dir


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
