"""
Configuration File for Forecasting Pipeline

Central place to configure all parameters for the forecasting system.
Modify values here to experiment with different settings.
"""

# ==============================================================================
# DATA GENERATION
# ==============================================================================
DATA_CONFIG = {
    'generate_new_data': True,  # Set False to use existing data
    'data_path': 'data/sku_day_data.csv',  # Path if using existing data
    'start_date': '2022-01-01',
    'n_days': 1095,  # Total days to generate (3 years for yearly seasonality)
    'categories': ['Beverages', 'Frozen_Foods', 'Bakery'],
    'skus_per_category': 10,
    'seed': 42
}

# ==============================================================================
# TRAIN/VAL/TEST SPLIT
# ==============================================================================
SPLIT_CONFIG = {
    'train_days': 1000,  # Training period (needs >=730 for yearly seasonality)
    'val_days': 30,      # Validation period
    'test_days': 30      # Test period
}

# ==============================================================================
# ETS MODEL CONFIGURATION
# ==============================================================================
ETS_CONFIG = {
    'seasonal_periods': 7,  # 365 for yearly, 7 for weekly seasonality
    'trend': 'add',           # 'add', 'mul', or None
    'seasonal': 'mul',        # 'add', 'mul', or None
    'damped_trend': True,     # Use damped trend

    # Smoothing parameters (set to None to let model optimize)
    'smoothing_level': None,     # Alpha (0-1, higher = more weight to recent)
    'smoothing_trend': None,      # Beta (0-1)
    'smoothing_seasonal': None    # Gamma (0-1)
}

# ==============================================================================
# LIGHTGBM MODEL CONFIGURATION
# ==============================================================================
LGBM_CONFIG = {
    # Training parameters
    'num_boost_round': 300,
    'early_stopping_rounds': 30,

    # Default model parameters (used if not tuning)
    'params': {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42
    }
}

# ==============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ==============================================================================
TUNING_CONFIG = {
    'tune_hyperparameters': True,  # Enable/disable hyperparameter tuning
    'n_trials': 50,                # Number of Optuna trials
    'cv_splits': 3,                # Time series CV splits
    'min_train_size': 500,         # Minimum training size for CV
    'val_size': 30,                # Validation size for each CV split
    'tuned_params_path': 'models/tuned_params.json'
}

# ==============================================================================
# FORECAST CONFIGURATION
# ==============================================================================
FORECAST_CONFIG = {
    'forecast_horizon': 30  # Number of days to forecast
}

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================
OUTPUT_CONFIG = {
    'output_dir': 'outputs',
    'model_dir': 'models',
    'data_dir': 'data'
}

# ==============================================================================
# AGGREGATION CONFIGURATION
# ==============================================================================
AGGREGATION_CONFIG = {
    'instock_threshold': 60.0,  # Minimum instock rate (%) to include SKU-day record
    'interpolation_method': 'linear',  # Method to fill gaps after filtering
    'interpolation_limit': 7  # Max consecutive days to interpolate (None = no limit)
}

# ==============================================================================
# FEATURE ENGINEERING (Advanced)
# ==============================================================================
FEATURE_CONFIG = {
    # Lag features (in days)
    'lag_periods': [7, 14, 28],  # Removed 364 to reduce data loss

    # Rolling window sizes (in days)
    'rolling_windows': [7, 28],  # Removed 364 to reduce data loss

    # Holiday window (days before/after)
    'holiday_pre_window': 7,
    'holiday_post_window': 7,

    # Weather extremes (percentiles)
    'temp_cold_percentile': 0.1,
    'temp_hot_percentile': 0.9,
    'precip_heavy_percentile': 0.9
}

# ==============================================================================
# EVALUATION METRICS TARGETS
# ==============================================================================
METRIC_TARGETS = {
    'mape_target': 20.0,  # Target MAPE < 20%
    'mase_target': 1.0    # Target MASE < 1.0 (better than naive)
}
