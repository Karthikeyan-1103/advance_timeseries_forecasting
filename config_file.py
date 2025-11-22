"""
Configuration file for Time Series Forecasting Project
Modify these settings to customize the model and training process
"""


# DATA CONFIGURATION


DATA_CONFIG = {
    # Synthetic data generation
    'n_samples': 1000,
    'start_date': '2020-01-01',
    'freq': 'D',  # Daily frequency
    
    # Data split
    'train_split': 0.8,
    
    # Time series characteristics
    'trend_coefficient': 0.05,
    'seasonality_periods': [50, 25],
    'seasonality_amplitudes': [10, 5],
    'noise_std': 2,
    'base_value': 50,
}

# MODEL CONFIGURATION

MODEL_CONFIG = {
    # Sequence parameters
    'lookback': 60,           
    'forecast_horizon': 10,  
    
    # Model architecture
    'lstm_units': [128, 64, 32],  
    'dense_units': [64, 32],       
    'dropout_rate': 0.2,
    'use_bidirectional': True,
    
    # Optimizer settings
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'huber',  
}

# TRAINING CONFIGURATION

TRAINING_CONFIG = {
    # Training parameters
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    
    # Callbacks
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 15,
        'restore_best_weights': True
    },
    
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-7
    },
    
    'model_checkpoint': {
        'monitor': 'val_loss',
        'save_best_only': True,
        'filepath': 'best_model.keras'
    }
}

# EVALUATION CONFIGURATION

EVALUATION_CONFIG = {
    # Metrics to compute
    'metrics': ['mse', 'mae', 'rmse', 'r2'],
    
    # Visualization settings
    'plot_samples': 5,
    'figure_size': (20, 12),
    'dpi': 300,
}

# FORECASTING CONFIGURATION

FORECAST_CONFIG = {
    # Future forecasting
    'future_steps': 30,
    'confidence_interval': 0.95,
}


# FILE PATHS


PATHS = {
    'data_dir': 'data/',
    'models_dir': 'models/',
    'results_dir': 'results/',
    'logs_dir': 'logs/',
    
    # Output files
    'trained_model': 'models/best_model.keras',
    'scaler': 'models/scaler.pkl',
    'results_plot': 'results/forecasting_results.png',
    'analysis_plot': 'results/time_series_analysis.png',
    'metrics_file': 'results/metrics.json',
    'predictions_file': 'results/predictions.csv',
}


# PREPROCESSING CONFIGURATION

PREPROCESSING_CONFIG = {
    # Scaling
    'scaler_type': 'minmax',  # Options: 'minmax', 'standard'
    'feature_range': (0, 1),
    
    # Missing values
    'fill_method': 'linear',  # Options: 'linear', 'ffill', 'bfill', 'mean'
    
    # Outlier detection
    'outlier_threshold': 3,  # Z-score threshold
    'remove_outliers': False,
    
    # Feature engineering
    'add_time_features': False,
    'add_rolling_stats': False,
    'rolling_windows': [7, 30, 90],
}


# ADVANCED MODEL CONFIGURATIONS

# Alternative model architectures you can experiment with

GRU_CONFIG = {
    'model_type': 'GRU',
    'gru_units': [128, 64, 32],
    'dropout_rate': 0.2,
    'use_bidirectional': True,
}

SIMPLE_RNN_CONFIG = {
    'model_type': 'SimpleRNN',
    'rnn_units': [64, 32],
    'dropout_rate': 0.3,
}

ATTENTION_LSTM_CONFIG = {
    'model_type': 'LSTM_Attention',
    'lstm_units': [128, 64],
    'attention_units': 64,
    'dropout_rate': 0.2,
}


# HYPERPARAMETER TUNING


TUNING_CONFIG = {
    'enable_tuning': False,
    
    'search_space': {
        'lookback': [30, 60, 90],
        'lstm_units': [[64, 32], [128, 64], [128, 64, 32]],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.1, 0.2, 0.3],
    },
    
    'tuning_method': 'random_search',  # Options: 'random_search', 'grid_search'
    'n_trials': 20,
}


# LOGGING CONFIGURATION


LOGGING_CONFIG = {
    'level': 'INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/training.log',
}


# VISUALIZATION CONFIGURATION


VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl',
    'font_size': 12,
    'line_width': 2,
    'figure_format': 'png',
}


# PRODUCTION CONFIGURATION


PRODUCTION_CONFIG = {
    'model_version': '1.0.0',
    'serve_model': False,
    'api_port': 5000,
    'max_batch_size': 128,
    'enable_monitoring': True,
}


def get_config():
    """
    Get complete configuration as a dictionary
    
    Returns:
        Dictionary with all configuration settings
    """
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'forecast': FORECAST_CONFIG,
        'paths': PATHS,
        'preprocessing': PREPROCESSING_CONFIG,
        'tuning': TUNING_CONFIG,
        'logging': LOGGING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'production': PRODUCTION_CONFIG,
    }


def print_config():
    """Print current configuration"""
    config = get_config()
    
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print_config()
