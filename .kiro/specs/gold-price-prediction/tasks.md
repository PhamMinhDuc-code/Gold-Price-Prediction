### Implementation Plan: Gold Price Prediction System

## Overview

This implementation plan breaks down the gold price prediction system into discrete, actionable coding tasks. The system will be implemented in Python using TensorFlow/Keras for deep learning models, scikit-learn and XGBoost for traditional ML models, and pandas for data processing. The tasks are organized to build incrementally from data handling through model training to prediction and evaluation.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure (data/, models/, reports/, src/, tests/)
  - Create requirements.txt with all necessary dependencies
  - Create config.py with system configuration parameters
  - Set up logging configuration
  - Create __init__.py files for Python package structure
  - _Requirements: All requirements (infrastructure setup)_

- [x] 2. Implement data ingestion and validation module
  - [ ] 2.1 Create DataIngestionManager class with CSV loading
    - Implement load_csv() method to read OHLCV data from CSV files
    - Parse dates and set as index
    - _Requirements: 1.1_
  
  - [ ] 2.2 Implement OHLCV data validation
    - Implement validate_ohlcv_data() to check required columns
    - Validate High >= Low constraint
    - Validate Close and Open within [Low, High] range
    - _Requirements: 1.2, 1.7, 1.8_
  
  - [ ] 2.3 Implement chronological and duplicate checking
    - Implement validate_chronological_order() method
    - Implement check_duplicates() method
    - _Requirements: 1.5, 1.6_
  
  - [ ] 2.4 Implement economic indicators fetching
    - Implement load_economic_indicators() using yfinance API
    - Fetch DXY, Oil prices, and Treasury yields
    - Validate economic indicator data structure
    - _Requirements: 1.3_
  
  - [ ] 2.5 Implement missing value detection
    - Add logic to flag missing values in OHLCV data
    - Return validation results with affected records
    - _Requirements: 1.4_
  
  - [x] 2.6 Write unit tests for data ingestion
    - Test CSV loading with valid and invalid files
    - Test OHLCV constraint validation (High < Low should fail)
    - Test chronological order validation
    - Test duplicate detection
    - Test missing value detection
    - _Requirements: 1.1-1.8_

- [x] 3. Implement data preprocessing and cleaning module
  - [ ] 3.1 Create DataPreprocessor class with missing value handling
    - Implement handle_missing_values() with forward-fill for gaps ≤3 days
    - Implement interpolate_economic_indicators() with linear interpolation
    - _Requirements: 2.1, 2.2_
  
  - [ ] 3.2 Implement normalization functionality
    - Implement normalize_features() with min-max and z-score options
    - Store scaling parameters for later inverse transformation
    - Return tuple of (normalized_df, scaling_params_dict)
    - _Requirements: 2.3_
  
  - [ ] 3.3 Implement dataset alignment
    - Implement align_datasets() to merge gold data with economic indicators
    - Use date-based inner join with forward-fill strategy
    - _Requirements: 2.4_
  
  - [ ] 3.4 Implement outlier removal
    - Implement remove_outliers() using z-score method (3 std threshold)
    - Log outlier statistics before removal
    - _Requirements: 2.5_
  
  - [ ] 3.5 Implement data quality reporting
    - Implement generate_quality_report() method
    - Calculate and return records processed, missing values handled, outliers removed
    - Calculate data quality score (0-100)
    - _Requirements: 2.6_
  
  - [x] 3.6 Write unit tests for preprocessing
    - Test forward-fill works correctly for gaps ≤3 days
    - Test min-max normalization produces values in [0, 1]
    - Test z-score standardization
    - Test outlier removal with known data
    - Test dataset alignment with mismatched dates
    - _Requirements: 2.1-2.6_

- [x] 4. Checkpoint - Verify data pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement feature engineering module
  - [ ] 5.1 Create FeatureEngineer class with lag features
    - Implement create_lag_features() for specified lag periods [1, 7, 14, 30]
    - Apply to Close price column
    - _Requirements: 3.1_
  
  - [ ] 5.2 Implement rolling statistics features
    - Implement create_rolling_features() for rolling mean and std
    - Use windows [7, 14, 30, 90] for mean and [7, 14, 30] for std
    - _Requirements: 3.2, 3.3_
  
  - [ ] 5.3 Implement technical indicators
    - Implement create_technical_indicators() method
    - Calculate RSI (14-day period)
    - Calculate MACD (12, 26, 9 parameters)
    - Calculate Bollinger Bands (20-day, 2 std)
    - Use ta-lib or pandas-ta library
    - _Requirements: 3.4_
  
  - [ ] 5.4 Implement interaction features
    - Implement create_interaction_features() method
    - Create Gold/Oil ratio
    - Create Gold/DXY rolling correlation (30-day)
    - Create Gold vs Treasury yield features
    - _Requirements: 3.5_
  
  - [ ] 5.5 Implement temporal features
    - Implement create_temporal_features() method
    - Extract day_of_week, month, quarter, year
    - Add boolean flags: is_quarter_end, is_year_end
    - _Requirements: 3.6_
  
  - [ ] 5.6 Implement complete feature set builder
    - Implement build_feature_set() to orchestrate all feature engineering
    - Call all feature creation methods in sequence
    - Handle NaN values created by lag/rolling operations
    - Return complete feature DataFrame
    - _Requirements: 3.7_
  
  - [ ] 5.7 Write unit tests for feature engineering
    - Test lag feature creation with known data
    - Test rolling mean and std calculations
    - Test RSI calculation matches expected values
    - Test MACD and Bollinger Bands
    - Test interaction feature calculations
    - Test temporal feature extraction
    - _Requirements: 3.1-3.7_

- [x] 6. Implement dataset splitting and preparation
  - [ ] 6.1 Create DatasetSplitter class with chronological splitting
    - Implement split_dataset() with 70/15/15 train/val/test ratios
    - Ensure chronological order (no shuffling)
    - _Requirements: 4.1, 4.2_
  
  - [ ] 6.2 Implement split integrity verification
    - Implement verify_split_integrity() method
    - Check no date overlap between splits
    - Verify minimum 100 records per subset
    - _Requirements: 4.3_
  
  - [ ] 6.3 Implement feature-target separation
    - Implement prepare_feature_target_split() method
    - Separate features (X) from target (y - Close price)
    - Return numpy arrays
    - _Requirements: 4.4_
  
  - [ ] 6.4 Implement sequence creation for LSTM/GRU
    - Implement create_sequences() method
    - Create sliding windows of length 60 for time series models
    - Return (X_sequences, y_targets) with proper shapes
    - _Requirements: 4.4_
  
  - [ ]* 6.5 Write unit tests for dataset preparation
    - Test chronological splitting with known data
    - Test minimum sample verification
    - Test feature-target separation
    - Test sequence creation produces correct shapes
    - _Requirements: 4.1-4.5_

- [x] 7. Implement model architectures
  - [ ] 7.1 Create ModelTrainingPipeline class structure
    - Initialize class with configuration parameters
    - Set up model storage directory
    - _Requirements: 5.1_
  
  - [ ] 7.2 Implement LSTM model builder
    - Implement build_lstm_model() using Keras Sequential API
    - Architecture: LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
    - Use Adam optimizer with configurable learning rate
    - _Requirements: 5.3_
  
  - [ ] 7.3 Implement GRU model builder
    - Implement build_gru_model() using Keras Sequential API
    - Architecture: GRU(128) → Dropout(0.2) → GRU(64) → Dropout(0.2) → Dense(32) → Dense(1)
    - Use Adam optimizer with configurable learning rate
    - _Requirements: 5.3_
  
  - [ ] 7.4 Implement XGBoost model builder
    - Implement build_xgboost_model() using XGBRegressor
    - Configure with hyperparameters (max_depth, n_estimators, learning_rate)
    - _Requirements: 5.3_
  
  - [ ] 7.5 Implement Random Forest model builder
    - Implement build_random_forest_model() using RandomForestRegressor
    - Configure with hyperparameters (n_estimators, max_depth, min_samples_split)
    - _Requirements: 5.3_
  
  - [ ] 7.6 Write unit tests for model builders
    - Test LSTM model builds with correct architecture
    - Test GRU model builds with correct architecture
    - Test XGBoost and Random Forest instantiation
    - Test models accept expected input shapes
    - _Requirements: 5.3_

- [ ] 8. Checkpoint - Verify model architecture setup
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement model training functionality
  - [ ] 9.1 Implement basic model training
    - Implement train_model() method for Keras models
    - Add early stopping callback (patience=10)
    - Add model checkpointing to save best model
    - Use validation set for monitoring
    - _Requirements: 5.1_
  
  - [ ] 9.2 Implement training for tree-based models
    - Extend train_model() to handle XGBoost and Random Forest
    - Use eval_set for XGBoost early stopping
    - _Requirements: 5.1_
  
  - [ ] 9.3 Implement training metrics logging
    - Implement log_training_metrics() method
    - Log training time, loss curves, convergence status
    - Save training history to JSON file
    - _Requirements: 5.6_
  
  - [ ] 9.4 Implement hyperparameter tuning
    - Implement hyperparameter_tuning() method
    - Use GridSearchCV for tree-based models
    - Use manual loop with different configs for deep learning models
    - Return best hyperparameters based on validation performance
    - _Requirements: 5.2, 5.4_
  
  - [ ] 9.5 Write integration tests for model training
    - Test LSTM training on small synthetic dataset
    - Test XGBoost training completes successfully
    - Test early stopping triggers correctly
    - Test training metrics are logged
    - _Requirements: 5.1-5.6_

- [x] 10. Implement model persistence and registry
  - [ ] 10.1 Create ModelRegistry class structure
    - Initialize with models directory path
    - Create registry.json file if not exists
    - _Requirements: 8.1_
  
  - [ ] 10.2 Implement model saving
    - Implement save_model() in ModelTrainingPipeline
    - Save Keras models as .h5 files
    - Save scikit-learn/XGBoost models using joblib
    - Save metadata as JSON (version, hyperparams, features, metrics)
    - Save scaling parameters as pickle file
    - _Requirements: 5.5, 8.1, 8.2, 8.3_
  
  - [ ] 10.3 Implement model loading
    - Implement load_model() in ModelRegistry
    - Load model file based on type
    - Load and parse metadata JSON
    - Load scaling parameters
    - _Requirements: 8.4_
  
  - [ ] 10.4 Implement model registry functionality
    - Implement register_model() to add model to registry.json
    - Implement list_models() to show all versions
    - Implement load_model_by_version() and load_latest_model()
    - Implement validate_model_compatibility() to check schema
    - _Requirements: 8.1, 8.4, 8.5_
  
  - [ ] 10.5 Write unit tests for model persistence
    - Test model save/load round-trip preserves predictions
    - Test metadata is correctly saved and loaded
    - Test model registry tracks versions correctly
    - Test compatibility validation works
    - _Requirements: 8.1-8.5_

- [X] 11. Implement prediction service
  - [ ] 11.1 Create PredictionService class structure
    - Initialize with model registry
    - Load trained model and metadata
    - _Requirements: 6.1_
  
  - [ ] 11.2 Implement single-step prediction
    - Implement predict_single_step() method
    - Accept preprocessed input features
    - Generate next-day prediction
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [ ] 11.3 Implement multi-step prediction
    - Implement predict_multi_step() method
    - Use recursive strategy (predictions as inputs for future steps)
    - Support horizons up to 30 days
    - _Requirements: 6.2, 6.3_
  
  - [ ] 11.4 Implement denormalization
    - Implement denormalize_predictions() method
    - Apply inverse transformation using saved scaling parameters
    - Convert predictions back to original price scale
    - _Requirements: 6.4_
  
  - [ ] 11.5 Implement prediction intervals
    - Implement compute_prediction_intervals() method
    - Use bootstrap or quantile regression for confidence intervals
    - Default to 95% confidence level
    - _Requirements: 6.6_
  
  - [ ] 11.6 Implement timestamped predictions
    - Implement predict_with_timestamps() method
    - Generate DataFrame with prediction dates and values
    - Include confidence intervals if requested
    - _Requirements: 6.5_
  
  - [ ]* 11.7 Write unit tests for prediction service
    - Test single-step prediction produces valid output
    - Test multi-step prediction generates correct number of forecasts
    - Test denormalization inverts normalization correctly
    - Test prediction intervals are reasonable
    - _Requirements: 6.1-6.6_

- [ ] 12. Checkpoint - Verify training and prediction pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement model evaluation module
  - [ ] 13.1 Create ModelEvaluator class structure
    - Initialize with test data and predictions
    - _Requirements: 7.1_
  
  - [ ] 13.2 Implement regression metrics
    - Implement calculate_mae() using sklearn or manual calculation
    - Implement calculate_rmse() using sklearn or manual calculation
    - Implement calculate_mape() with proper handling of zero values
    - Implement calculate_r2() using sklearn
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [ ] 13.3 Implement directional accuracy
    - Implement calculate_directional_accuracy() method
    - Compare predicted vs actual price movement direction (up/down)
    - Return percentage of correct direction predictions
    - _Requirements: 7.6_
  
  - [ ] 13.4 Implement residual plotting
    - Implement plot_residuals() method
    - Create residual plot (prediction errors over time)
    - Save plot as PNG file
    - _Requirements: 7.5_
  
  - [ ] 13.5 Implement prediction vs actual plotting
    - Implement plot_predictions_vs_actual() method
    - Create line plot comparing predicted and actual values
    - Save plot as PNG file
    - _Requirements: 7.5_
  
  - [ ] 13.6 Implement performance report generation
    - Implement generate_performance_report() method
    - Compile all metrics and plots into comprehensive report
    - Save report as JSON and PDF/HTML
    - _Requirements: 7.7_
  
  - [ ]* 13.7 Write unit tests for evaluation metrics
    - Test MAE calculation with known values
    - Test RMSE calculation
    - Test MAPE calculation handles edge cases
    - Test R² calculation
    - Test directional accuracy counting
    - _Requirements: 7.1-7.7_

- [x] 14. Implement visualization and reporting module
  - [ ] 14.1 Create VisualizationManager class structure
    - Initialize with matplotlib/seaborn configuration
    - Set up consistent styling for all plots
    - _Requirements: 9.1_
  
  - [ ] 14.2 Implement time series prediction plotting
    - Implement plot_time_series_with_predictions() method
    - Plot actual prices, predictions, and confidence intervals
    - Use different colors/styles for clarity
    - _Requirements: 9.1, 9.2_
  
  - [ ] 14.3 Implement feature importance plotting
    - Implement plot_feature_importance() method
    - Extract feature importance from tree-based models
    - Create horizontal bar chart
    - _Requirements: 9.3_
  
  - [ ] 14.4 Implement indicator overlay plotting
    - Implement plot_indicators_overlay() method
    - Create multi-panel plot with gold prices and economic indicators
    - Use secondary y-axes where appropriate
    - _Requirements: 9.4_
  
  - [ ] 14.5 Implement training history plotting
    - Implement plot_training_history() method
    - Plot training and validation loss curves
    - Highlight best epoch
    - _Requirements: 5.6_
  
  - [ ] 14.6 Implement comprehensive prediction report
    - Implement create_prediction_report() method
    - Compile predictions, metrics, and plots into single report
    - Export as HTML or PDF
    - _Requirements: 9.5_
  
  - [ ]* 14.7 Write integration tests for visualization
    - Test plots are created without errors
    - Test files are saved to correct locations
    - Test report generation includes all sections
    - _Requirements: 9.1-9.5_

- [x] 15. Implement error handling and monitoring
  - [ ] 15.1 Define custom exception classes
    - Create DataValidationError, MissingColumnError, ChronologicalOrderError
    - Create ConstraintViolationError for OHLC constraint violations
    - Create PredictionError, ModelLoadError, ExtrapolationWarning
    - _Requirements: 10.1_
  
  - [ ] 15.2 Add validation error handling to DataIngestionManager
    - Raise appropriate exceptions with descriptive messages
    - Include specific details about failed validation checks
    - _Requirements: 10.1_
  
  - [ ] 15.3 Implement quality monitoring in DataPreprocessor
    - Check if missing values exceed 20% threshold
    - Log warning and flag dataset as low quality
    - _Requirements: 10.3_
  
  - [ ] 15.4 Add extrapolation warnings to PredictionService
    - Check if input features are outside training data range
    - Log ExtrapolationWarning with feature name and range
    - _Requirements: 10.2_
  
  - [ ] 15.5 Implement model load error handling
    - Wrap model loading in try-except blocks
    - Return ModelLoadError with file path and reason
    - _Requirements: 10.4_
  
  - [ ] 15.6 Implement prediction drift detection
    - Create QualityMonitor class with detect_prediction_drift() method
    - Compare current metrics against baseline
    - Trigger alert if error increases >25%
    - _Requirements: 10.5, 10.6_
  
  - [ ] 15.7 Write unit tests for error handling
    - Test custom exceptions are raised correctly
    - Test data quality threshold detection
    - Test extrapolation warnings are logged
    - Test drift detection triggers correctly
    - _Requirements: 10.1-10.6_

- [x] 16. Create end-to-end pipeline orchestration
  - [ ] 16.1 Create main pipeline script
    - Create main.py to orchestrate full pipeline
    - Accept command-line arguments for configuration
    - _Requirements: All requirements (integration)_
  
  - [ ] 16.2 Implement training pipeline mode
    - Load data → preprocess → engineer features → split → train → evaluate
    - Save trained model with metadata
    - Generate training report
    - _Requirements: 1.1-5.6, 7.1-7.7_
  
  - [ ] 16.3 Implement prediction pipeline mode
    - Load trained model → accept new data → generate predictions → visualize
    - Support both single and multi-step predictions
    - Generate prediction report
    - _Requirements: 6.1-6.6, 9.1-9.5_
  
  - [ ] 16.4 Implement evaluation pipeline mode
    - Load model and test data → generate predictions → evaluate → report
    - Compare multiple model versions
    - _Requirements: 7.1-7.7, 9.1-9.5_
  
  - [ ]* 16.5 Write integration tests for end-to-end pipeline
    - Test complete training pipeline with sample data
    - Test complete prediction pipeline produces valid forecasts
    - Test evaluation pipeline generates all metrics
    - _Requirements: All requirements_

- [ ] 17. Create example notebooks and documentation
  - [ ] 17.1 Create Jupyter notebook for data exploration
    - Load and visualize gold price data
    - Show economic indicators correlation with gold
    - Demonstrate data quality checks
    - _Requirements: 1.1-2.6_
  
  - [ ] 17.2 Create Jupyter notebook for model training
    - Demonstrate complete training workflow
    - Show hyperparameter tuning results
    - Compare different model architectures
    - _Requirements: 5.1-5.6_
  
  - [ ] 17.3 Create Jupyter notebook for prediction
    - Load trained model and generate predictions
    - Visualize predictions with confidence intervals
    - Demonstrate different forecast horizons
    - _Requirements: 6.1-6.6, 9.1-9.5_
  
  - [ ] 17.4 Write README.md documentation
    - Project overview and objectives
    - Installation instructions
    - Usage examples for training and prediction
    - Configuration guide
    - _Requirements: All requirements (documentation)_

- [ ] 18. Final checkpoint - Complete system validation
  - Run complete end-to-end pipeline on full historical dataset
  - Verify all models train successfully
  - Generate and review comprehensive evaluation reports
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional testing tasks that can be skipped for faster implementation
- Each task references specific requirements for traceability
- The implementation builds incrementally: data handling → feature engineering → model training → prediction → evaluation
- Multiple checkpoints ensure validation at key stages
- Unit tests validate individual components, integration tests validate workflows
- The system supports multiple model types (LSTM, GRU, XGBoost, Random Forest) for comparison
- All models include comprehensive evaluation metrics and visualizations
- Error handling and monitoring ensure robust production operation
