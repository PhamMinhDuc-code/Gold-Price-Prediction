# Gold Price Prediction System

A comprehensive machine learning system for forecasting gold (XAU) prices using historical OHLCV data and economic indicators. The system implements end-to-end ML pipelines including data ingestion, preprocessing, feature engineering, model training (LSTM, GRU, XGBoost, Random Forest), prediction generation, and performance evaluation.

## Project Structure

```
Gold Prediction/
├── data/               # Raw and processed data files
├── models/             # Trained model files and metadata
│   ├── checkpoints/    # Training checkpoints
│   └── registry.json   # Model version registry
├── reports/            # Evaluation reports and visualizations
├── notebooks/          # Jupyter notebooks for exploration and examples
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_prediction_and_forecasting.ipynb
├── src/                # Source code modules
│   ├── data_ingestion.py          # Data loading and validation
│   ├── data_preprocessing.py      # Data cleaning and normalization
│   ├── feature_engineering.py     # Feature creation and transformations
│   ├── dataset_splitter.py        # Train/val/test splitting
│   ├── model_training.py          # Model training pipeline
│   ├── prediction_service.py      # Prediction generation
│   ├── model_evaluator.py         # Performance metrics
│   ├── model_registry.py          # Model versioning
│   ├── visualization_manager.py   # Plotting and reporting
│   ├── quality_monitor.py         # Data quality monitoring
│   ├── exceptions.py              # Custom exceptions
│   ├── logger.py                  # Logging configuration
│   └── __init__.py
├── tests/              # Unit and integration tests
│   ├── test_data_ingestion.py
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── ...
├── logs/               # Application log files
├── config.py           # System configuration parameters
├── requirements.txt    # Python dependencies
├── main.py             # Main entry point
└── README.md          # Project documentation
```

## Features

### Data Processing
- **Data Ingestion**: Load OHLCV data from CSV and economic indicators (DXY, Oil, Treasury yields) via yfinance API
- **Data Validation**: Comprehensive checks for OHLC constraints, chronological order, missing values
- **Data Preprocessing**: Handle missing values with forward-fill/interpolation, normalize features, remove outliers
- **Feature Engineering**: Create lag features, rolling statistics, technical indicators (RSI, MACD, Bollinger Bands), interaction features

### Model Training
- **Multiple Architectures**: Support for LSTM, GRU, XGBoost, and Random Forest models
- **Hyperparameter Tuning**: Grid search and random search optimization
- **Time Series Splitting**: Proper train/validation/test splits preventing data leakage
- **Model Versioning**: Complete model registry with metadata and performance tracking

### Prediction & Evaluation
- **Prediction Service**: Generate single-step and multi-step forecasts (up to 30 days)
- **Confidence Intervals**: Uncertainty quantification at configurable confidence levels
- **Comprehensive Metrics**: MAE, RMSE, MAPE, R², directional accuracy
- **Visualization**: Time series plots, feature importance, residuals, comprehensive HTML/JSON reports

### Quality & Monitoring
- **Data Quality Reports**: Automated quality scoring and anomaly detection
- **Prediction Drift Detection**: Monitor model performance degradation over time
- **Logging**: Comprehensive logging for all components
- **Testing**: Full test suite with unit and integration tests

## Installation

### Prerequisites
- Python 3.8 or higher (tested on Python 3.8-3.12)
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone or download the repository**
```bash
cd "Gold Prediction"
```

2. **Create a virtual environment**
```bash
python -m venv .venv
```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Verify installation**
```bash
python verify_setup.py
```

### Dependencies

The system uses the following major libraries:
- **keras** (≥3.0.0) with JAX backend for deep learning
- **scikit-learn** (≥1.2.0) for ML algorithms and metrics
- **xgboost** (≥1.7.0) for gradient boosting
- **pandas** (≥1.5.0) and **numpy** (≥1.23.0) for data processing
- **yfinance** (≥0.2.0) for economic indicator data
- **matplotlib** (≥3.6.0) and **seaborn** (≥0.12.0) for visualization
- **pandas-ta** (≥0.3.0) for technical indicators

See `requirements.txt` for complete dependency list.

## Configuration

All system parameters are centralized in `config.py`. Key configuration categories:

### Directory Paths
```python
DATA_DIR = "data/"          # Data files
MODEL_DIR = "models/"       # Trained models
REPORTS_DIR = "reports/"    # Output reports
LOGS_DIR = "logs/"          # Log files
```

### Model Training Parameters
```python
SEQUENCE_LENGTH = 60        # Time steps for LSTM/GRU
TRAIN_RATIO = 0.7          # Training set proportion
VAL_RATIO = 0.15           # Validation set proportion
TEST_RATIO = 0.15          # Test set proportion
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
```

### Feature Engineering
```python
LAG_PERIODS = [1, 7, 14, 30]              # Lag days
ROLLING_WINDOWS = [7, 14, 30, 90]         # Rolling mean windows
RSI_PERIOD = 14                            # RSI indicator period
MACD_FAST = 12                             # MACD fast period
MACD_SLOW = 26                             # MACD slow period
BOLLINGER_WINDOW = 20                      # Bollinger bands window
```

### Economic Indicators
```python
INDICATORS = {
    'DXY': 'DX-Y.NYB',      # US Dollar Index
    'Oil': 'CL=F',          # Crude Oil Futures
    'Treasury_10Y': '^TNX'  # 10-Year Treasury Yield
}
```

### Quality Monitoring
```python
MAX_MISSING_PCT = 0.20          # 20% missing value threshold
DRIFT_THRESHOLD = 0.25           # 25% error increase triggers alert
OUTLIER_STD_THRESHOLD = 3.0      # Outlier detection (std devs)
```

### Prediction Parameters
```python
DEFAULT_FORECAST_HORIZON = 30    # Default prediction days
CONFIDENCE_LEVEL = 0.95          # Confidence interval level
```

Modify these parameters in `config.py` to customize system behavior.

## Usage

### Quick Start with Jupyter Notebooks

The easiest way to get started is through the provided Jupyter notebooks:

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`)
   - Load and validate gold price data
   - Visualize price trends and distributions
   - Analyze correlations with economic indicators
   - Assess data quality

2. **Model Training** (`notebooks/02_model_training.ipynb`)
   - Complete training workflow
   - Train multiple model architectures (LSTM, XGBoost)
   - Compare model performance
   - Hyperparameter tuning examples

3. **Prediction & Forecasting** (`notebooks/03_prediction_and_forecasting.ipynb`)
   - Load trained models
   - Generate predictions with confidence intervals
   - Visualize forecasts
   - Create prediction reports

### Command-Line Usage

#### 1. Data Ingestion and Preprocessing

```python
from src.data_ingestion import DataIngestionManager
from src.data_preprocessing import DataPreprocessor
from config import Config

# Load gold price data
data_manager = DataIngestionManager()
gold_df = data_manager.load_csv('XAU_1d_data.csv')

# Validate data
validation_result = data_manager.validate_ohlcv_data(gold_df)
print(f"Validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")

# Load economic indicators
indicators = data_manager.load_economic_indicators(
    list(Config.INDICATORS.values()),
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Preprocess data
preprocessor = DataPreprocessor()
gold_df = preprocessor.handle_missing_values(gold_df)
gold_df = preprocessor.remove_outliers(gold_df)
combined_df = preprocessor.align_datasets(gold_df, indicators)

# Normalize features
normalized_df, scaling_params = preprocessor.normalize_features(
    combined_df, method='minmax'
)
```

#### 2. Feature Engineering

```python
from src.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer()

# Create all features
featured_df = feature_engineer.create_lag_features(
    combined_df, 'Close', Config.LAG_PERIODS
)
featured_df = feature_engineer.create_rolling_features(
    featured_df, 'Close', Config.ROLLING_WINDOWS
)
featured_df = feature_engineer.create_technical_indicators(featured_df)
featured_df = feature_engineer.create_temporal_features(featured_df)
featured_df = feature_engineer.create_interaction_features(featured_df)

# Drop rows with NaN values
featured_df = featured_df.dropna()
```

#### 3. Model Training

```python
from src.dataset_splitter import DatasetSplitter
from src.model_training import ModelTrainingPipeline

# Split dataset
splitter = DatasetSplitter()
train_df, val_df, test_df = splitter.split_dataset(
    featured_df,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Prepare sequences for LSTM
X_train, y_train = splitter.create_sequences(
    train_df.values, 
    Config.SEQUENCE_LENGTH,
    target_column_idx=train_df.columns.get_loc('Close')
)
X_val, y_val = splitter.create_sequences(val_df.values, Config.SEQUENCE_LENGTH, ...)

# Train LSTM model
pipeline = ModelTrainingPipeline()
lstm_model = pipeline.build_lstm_model(
    X_train.shape[1:],
    {'units_layer1': 128, 'units_layer2': 64, 'dropout': 0.2, 'learning_rate': 0.001}
)
result = pipeline.train_model(lstm_model, X_train, y_train, X_val, y_val)

print(f"Training completed in {result.training_time:.2f}s")
print(f"Final validation loss: {result.validation_loss:.6f}")
```

#### 4. Model Evaluation

```python
from src.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# Generate predictions
predictions = result.model.predict(X_test)

# Calculate metrics
mae = evaluator.calculate_mae(y_test, predictions)
rmse = evaluator.calculate_rmse(y_test, predictions)
mape = evaluator.calculate_mape(y_test, predictions)
r2 = evaluator.calculate_r2(y_test, predictions)
dir_acc = evaluator.calculate_directional_accuracy(y_test, predictions)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R²: {r2:.4f}")
print(f"Directional Accuracy: {dir_acc:.2%}")

# Generate performance report
report = evaluator.generate_performance_report(
    y_test, predictions, test_dates
)
```

#### 5. Save and Register Model

```python
from src.model_registry import ModelRegistry, ModelMetadata
from datetime import datetime

# Create metadata
metadata = ModelMetadata(
    version='v1.0.0',
    model_type='LSTM',
    training_date=datetime.now(),
    hyperparameters={'units_layer1': 128, 'units_layer2': 64, 'dropout': 0.2},
    feature_list=list(featured_df.columns),
    scaling_params=scaling_params,
    performance_metrics={'mae': mae, 'rmse': rmse, 'r2': r2},
    training_data_range=(train_df['Date'].min(), train_df['Date'].max()),
    sequence_length=Config.SEQUENCE_LENGTH
)

# Save model
model_path = pipeline.save_model(result.model, metadata, 'v1.0.0')

# Register in registry
registry = ModelRegistry()
registry.register_model(model_path, metadata, 'v1.0.0')
```

#### 6. Generate Predictions

```python
from src.prediction_service import PredictionService

# Load model
predictor = PredictionService()
model, metadata = registry.load_latest_model()

# Single-step prediction
prediction = predictor.predict_single_step(model, input_features)

# Multi-step prediction (30 days)
predictions = predictor.predict_multi_step(
    model, input_features, horizon=30, model_type=metadata.model_type
)

# With confidence intervals
confidence_intervals = predictor.compute_prediction_intervals(
    predictions, confidence=0.95
)

# Create predictions with timestamps
from datetime import timedelta
last_date = featured_df['Date'].iloc[-1]
dates = [last_date + timedelta(days=i+1) for i in range(30)]

forecast_df = pd.DataFrame({
    'Date': dates,
    'Predicted_Close': predictions,
    'Lower_CI': confidence_intervals[:, 0],
    'Upper_CI': confidence_intervals[:, 1]
})
```

#### 7. Visualization and Reporting

```python
from src.visualization_manager import VisualizationManager

viz_manager = VisualizationManager()

# Create comprehensive prediction report
report_path = viz_manager.create_prediction_report(
    predictions=forecast_df,
    actual=historical_df,
    confidence_intervals=confidence_intervals,
    metadata=metadata,
    output_format='html',
    report_name='gold_forecast_report'
)

print(f"Report saved to: {report_path}")

# Individual plots
viz_manager.plot_time_series_with_predictions(
    actual=historical_df,
    predictions=forecast_df,
    confidence_intervals=confidence_intervals
)

viz_manager.plot_feature_importance(model, feature_names)
```

### Using the Main Pipeline

For a complete end-to-end pipeline, use `main.py`:

```bash
python main.py
```

This will execute the full workflow: data loading → preprocessing → feature engineering → training → evaluation → prediction.

## Technology Stack

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning models (LSTM, GRU)
- **scikit-learn**: Traditional ML models and metrics
- **XGBoost**: Gradient boosting model
- **pandas/numpy**: Data processing
- **yfinance**: Economic indicator data download
- **matplotlib/seaborn**: Visualization
- **pandas-ta**: Technical indicators

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test module:
```bash
pytest tests/test_data_ingestion.py -v
```

## Logging

Logs are automatically created in the `logs/` directory. Configure logging in `src/logger.py` or through Config class:

```python
from src.logger import get_logger

logger = get_logger(__name__)
logger.info('Processing started')
```

## Model Versioning

Trained models are stored in `models/` with version identifiers:
```
models/
├── model_v1.0.0/
│   ├── model.h5          # Keras model
│   ├── metadata.json     # Training metadata
│   ├── scaler.pkl        # Normalization parameters
│   └── feature_list.json # Feature names
└── registry.json         # Model registry
```

## Development Roadmap

- [x] Project structure and configuration setup
- [ ] Data ingestion and validation module
- [ ] Data preprocessing and cleaning module
- [ ] Feature engineering module
- [ ] Model training pipeline
- [ ] Prediction service
- [ ] Model evaluation and visualization
- [ ] End-to-end pipeline orchestration

## License

Copyright © 2024 Gold Prediction System Team

## Contributors

Gold Prediction System Development Team
