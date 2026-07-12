"""
Model Registry Module

This module implements the ModelRegistry class for managing model versions,
persistence, and loading. It tracks all trained models and their metadata.

Requirements: 8.1, 8.4, 8.5
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import keras
import joblib
import pickle

from config import Config
from src.exceptions import ModelLoadError


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

# Add handler if not already present
if not logger.handlers:
    log_file = Config.get_log_path('src.model_registry.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class ModelInfo:
    """Container for model information in registry."""
    
    def __init__(self, version: str, model_type: str, training_date: datetime,
                 path: str, performance_metrics: Dict[str, float]):
        self.version = version
        self.model_type = model_type
        self.training_date = training_date
        self.path = path
        self.performance_metrics = performance_metrics
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'version': self.version,
            'model_type': self.model_type,
            'training_date': self.training_date.isoformat() if isinstance(self.training_date, datetime) else str(self.training_date),
            'path': self.path,
            'performance_metrics': self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelInfo':
        """Create ModelInfo from dictionary."""
        training_date = datetime.fromisoformat(data['training_date']) if isinstance(data['training_date'], str) else data['training_date']
        return cls(
            version=data['version'],
            model_type=data['model_type'],
            training_date=training_date,
            path=data['path'],
            performance_metrics=data['performance_metrics']
        )


class ModelRegistry:
    """
    Model Registry for managing model versions and persistence.
    
    This class provides functionality to:
    - Register new model versions
    - Load models by version or get latest
    - List all available models
    - Validate model compatibility
    
    Requirements: 8.1, 8.4, 8.5
    """
    
    def __init__(self, models_directory: Optional[Path] = None):
        """
        Initialize ModelRegistry with models directory path.
        
        Args:
            models_directory: Path to models directory. Defaults to Config.MODEL_DIR.
        
        Requirements: 8.1
        """
        self.models_dir = models_directory if models_directory is not None else Config.MODEL_DIR
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        self.registry_file = self.models_dir / "registry.json"
        
        # Create registry.json if it doesn't exist
        if not self.registry_file.exists():
            self._initialize_registry()
            logger.info(f"Created new registry file at: {self.registry_file}")
        else:
            logger.info(f"Using existing registry file at: {self.registry_file}")
        
        logger.info(f"ModelRegistry initialized with directory: {self.models_dir}")
    
    def _initialize_registry(self) -> None:
        """
        Create registry.json file if it doesn't exist.
        
        Requirements: 8.1
        """
        registry_data = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'models': []
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info("Initialized new model registry")
    
    def _load_registry(self) -> Dict:
        """Load registry data from file."""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}", exc_info=True)
            raise
    
    def _save_registry(self, registry_data: Dict) -> None:
        """Save registry data to file."""
        try:
            registry_data['last_updated'] = datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            logger.debug("Registry saved successfully")
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}", exc_info=True)
            raise
    
    def register_model(self, model_path: str, metadata: Dict, version: str) -> str:
        """
        Add model to registry.json.
        
        Args:
            model_path: Path to the model directory
            metadata: Model metadata dictionary
            version: Version string for the model
        
        Returns:
            Registry ID for the model
        
        Requirements: 8.1
        """
        logger.info(f"Registering model version {version}")
        
        # Load existing registry
        registry = self._load_registry()
        
        # Check if version already exists
        existing_versions = [m['version'] for m in registry['models']]
        if version in existing_versions:
            logger.warning(f"Model version {version} already exists in registry. Updating entry.")
            # Remove existing entry
            registry['models'] = [m for m in registry['models'] if m['version'] != version]
        
        # Create model entry
        model_entry = {
            'version': version,
            'model_type': metadata.get('model_type', 'unknown'),
            'training_date': metadata.get('training_date', datetime.now().isoformat()),
            'path': str(model_path),
            'performance_metrics': metadata.get('performance_metrics', {}),
            'registered_at': datetime.now().isoformat()
        }
        
        # Add to registry
        registry['models'].append(model_entry)
        
        # Save updated registry
        self._save_registry(registry)
        
        logger.info(f"Model version {version} registered successfully")
        logger.info(f"Total models in registry: {len(registry['models'])}")
        
        return version
    
    def list_models(self) -> List[ModelInfo]:
        """
        List all registered models with performance metrics.
        
        Returns:
            List of ModelInfo objects
        
        Requirements: 8.1
        """
        logger.info("Listing all registered models")
        
        registry = self._load_registry()
        models = []
        
        for model_data in registry['models']:
            try:
                model_info = ModelInfo.from_dict(model_data)
                models.append(model_info)
            except Exception as e:
                logger.warning(f"Error parsing model entry: {str(e)}")
                continue
        
        logger.info(f"Found {len(models)} registered models")
        return models
    
    def load_model(self, model_path: Path, model_type: str) -> Any:
        """
        Load model file based on type.
        
        Args:
            model_path: Path to model directory
            model_type: Type of model ('LSTM', 'GRU', 'XGBoost', 'RandomForest')
        
        Returns:
            Loaded model object
        
        Requirements: 8.4
        """
        logger.info(f"Loading {model_type} model from {model_path}")
        
        model_path = Path(model_path)
        
        try:
            if model_type in ['LSTM', 'GRU']:
                # Load Keras model (try .keras first, then .h5 for backward compatibility)
                model_file = model_path / 'model.keras'
                if not model_file.exists():
                    model_file = model_path / 'model.h5'
                
                if not model_file.exists():
                    error_msg = f"Model file not found: {model_file}"
                    logger.error(error_msg)
                    # Raise ModelLoadError with file path and reason (Requirement 10.4)
                    raise ModelLoadError(str(model_path), error_msg)
                
                model = keras.models.load_model(model_file)
                logger.info(f"Keras model loaded successfully from {model_file}")
                
            else:  # XGBoost, RandomForest, or other scikit-learn models
                # Load using joblib
                model_file = model_path / 'model.pkl'
                if not model_file.exists():
                    error_msg = f"Model file not found: {model_file}"
                    logger.error(error_msg)
                    # Raise ModelLoadError with file path and reason (Requirement 10.4)
                    raise ModelLoadError(str(model_path), error_msg)
                
                model = joblib.load(model_file)
                logger.info(f"Scikit-learn/XGBoost model loaded successfully from {model_file}")
            
            return model
            
        except ModelLoadError:
            # Re-raise ModelLoadError as-is
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            # Wrap other exceptions in ModelLoadError (Requirement 10.4)
            raise ModelLoadError(str(model_path), f"Unexpected error: {str(e)}")
    
    def load_metadata(self, model_path: Path) -> Dict:
        """
        Load and parse metadata JSON.
        
        Args:
            model_path: Path to model directory
        
        Returns:
            Metadata dictionary
        
        Requirements: 8.4
        """
        logger.info(f"Loading metadata from {model_path}")
        
        model_path = Path(model_path)
        metadata_file = model_path / 'metadata.json'
        
        try:
            if not metadata_file.exists():
                error_msg = f"Metadata file not found: {metadata_file}"
                logger.error(error_msg)
                # Raise ModelLoadError (Requirement 10.4)
                raise ModelLoadError(str(model_path), error_msg)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info("Metadata loaded successfully")
            logger.debug(f"Metadata keys: {list(metadata.keys())}")
            
            return metadata
            
        except ModelLoadError:
            # Re-raise ModelLoadError as-is
            raise
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}", exc_info=True)
            # Wrap other exceptions in ModelLoadError (Requirement 10.4)
            raise ModelLoadError(str(model_path), f"Failed to parse metadata: {str(e)}")
    
    def load_scaling_params(self, model_path: Path) -> Dict:
        """
        Load scaling parameters.
        
        Args:
            model_path: Path to model directory
        
        Returns:
            Scaling parameters dictionary
        
        Requirements: 8.4
        """
        logger.info(f"Loading scaling parameters from {model_path}")
        
        model_path = Path(model_path)
        scaler_file = model_path / 'scaler.pkl'
        
        try:
            if not scaler_file.exists():
                logger.warning(f"Scaler file not found: {scaler_file}. Returning empty dict.")
                return {}
            
            with open(scaler_file, 'rb') as f:
                scaling_params = pickle.load(f)
            
            logger.info("Scaling parameters loaded successfully")
            
            return scaling_params
            
        except Exception as e:
            logger.error(f"Error loading scaling parameters: {str(e)}", exc_info=True)
            raise
    
    def load_model_by_version(self, version: str) -> Tuple[Any, Dict]:
        """
        Load specific model version with metadata.
        
        Args:
            version: Model version string
        
        Returns:
            Tuple of (model, metadata)
        
        Requirements: 8.4
        """
        logger.info(f"Loading model by version: {version}")
        
        registry = self._load_registry()
        
        # Find model entry
        model_entry = None
        for entry in registry['models']:
            if entry['version'] == version:
                model_entry = entry
                break
        
        if model_entry is None:
            raise ValueError(f"Model version {version} not found in registry")
        
        model_path = Path(model_entry['path'])
        model_type = model_entry['model_type']
        
        # Load model
        model = self.load_model(model_path, model_type)
        
        # Load metadata
        metadata = self.load_metadata(model_path)
        
        # Load scaling parameters
        scaling_params = self.load_scaling_params(model_path)
        metadata['scaling_params'] = scaling_params
        
        logger.info(f"Model version {version} loaded successfully")
        
        return model, metadata
    
    def load_latest_model(self) -> Tuple[Any, Dict]:
        """
        Load most recent model version.
        
        Returns:
            Tuple of (model, metadata)
        
        Requirements: 8.4
        """
        logger.info("Loading latest model")
        
        registry = self._load_registry()
        
        if not registry['models']:
            raise ValueError("No models found in registry")
        
        # Sort by training_date and get latest
        sorted_models = sorted(
            registry['models'],
            key=lambda m: m.get('training_date', ''),
            reverse=True
        )
        
        latest_entry = sorted_models[0]
        latest_version = latest_entry['version']
        
        logger.info(f"Latest model version: {latest_version}")
        
        return self.load_model_by_version(latest_version)
    
    def validate_model_compatibility(self, model_metadata: Dict) -> bool:
        """
        Verify model is compatible with current data schema.
        
        Checks:
        - Feature list compatibility
        - Required metadata fields present
        - Model type is supported
        
        Args:
            model_metadata: Metadata dictionary to validate
        
        Returns:
            True if compatible, False otherwise
        
        Requirements: 8.5
        """
        logger.info("Validating model compatibility")
        
        # Check required fields
        required_fields = ['version', 'model_type', 'feature_list', 'scaling_params']
        missing_fields = [field for field in required_fields if field not in model_metadata]
        
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        # Check model type is supported
        supported_types = ['LSTM', 'GRU', 'XGBoost', 'RandomForest']
        model_type = model_metadata.get('model_type')
        
        if model_type not in supported_types:
            logger.error(f"Unsupported model type: {model_type}. Supported types: {supported_types}")
            return False
        
        # Check feature list is not empty
        feature_list = model_metadata.get('feature_list', [])
        if not feature_list:
            logger.error("Feature list is empty")
            return False
        
        # For sequence models, check sequence_length is present
        if model_type in ['LSTM', 'GRU']:
            if 'sequence_length' not in model_metadata:
                logger.error(f"Sequence length missing for {model_type} model")
                return False
        
        logger.info("Model compatibility validation passed")
        return True


if __name__ == "__main__":
    # Example usage
    print("ModelRegistry module loaded successfully")
    
    # Initialize registry
    registry = ModelRegistry()
    
    # List models
    models = registry.list_models()
    print(f"\nTotal models in registry: {len(models)}")
    
    for model_info in models:
        print(f"\nVersion: {model_info.version}")
        print(f"Type: {model_info.model_type}")
        print(f"Training Date: {model_info.training_date}")
        print(f"Performance: {model_info.performance_metrics}")
