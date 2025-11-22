"""
Utility functions for time series forecasting project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_csv_data(filepath, date_column='date', value_column='value', parse_dates=True):
    """
    Load time series data from CSV file
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        value_column: Name of value column
        parse_dates: Whether to parse dates
    
    Returns:
        DataFrame with date and value columns
    """
    try:
        df = pd.read_csv(filepath)
        
        if parse_dates and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
        
        if value_column not in df.columns:
            # Try to find numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_column = numeric_cols[0]
                print(f"⚠️ Using column '{value_column}' as value column")
        
        return df[[date_column, value_column]].rename(
            columns={date_column: 'date', value_column: 'value'}
        )
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def detect_anomalies(data, threshold=3):
    """
    Detect anomalies using Z-score method
    
    Args:
        data: Array of values
        threshold: Z-score threshold (default: 3)
    
    Returns:
        Boolean array indicating anomalies
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold


def fill_missing_values(df, method='linear'):
    """
    Fill missing values in time series
    
    Args:
        df: DataFrame with 'value' column
        method: Interpolation method ('linear', 'ffill', 'bfill', 'mean')
    
    Returns:
        DataFrame with filled values
    """
    if method == 'mean':
        df['value'].fillna(df['value'].mean(), inplace=True)
    elif method in ['ffill', 'bfill']:
        df['value'].fillna(method=method, inplace=True)
    else:
        df['value'].interpolate(method=method, inplace=True)
    
    return df


def add_time_features(df):
    """
    Add time-based features to DataFrame
    
    Args:
        df: DataFrame with 'date' column
    
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['dayofyear'] = df['date'].dt.dayofyear
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    return df


def calculate_rolling_statistics(df, windows=[7, 30, 90]):
    """
    Calculate rolling statistics
    
    Args:
        df: DataFrame with 'value' column
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling statistics
    """
    df = df.copy()
    
    for window in windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window=window).max()
    
    return df


def decompose_time_series(data, period=12):
    """
    Decompose time series into trend, seasonal, and residual components
    
    Args:
        data: Array of values
        period: Seasonal period
    
    Returns:
        Dictionary with trend, seasonal, and residual components
    """
    from scipy import signal
    
    # Trend (using moving average)
    trend = pd.Series(data).rolling(window=period, center=True).mean().values
    
    # Detrended
    detrended = data - trend
    
    # Seasonal (using FFT)
    fft = np.fft.fft(detrended[~np.isnan(detrended)])
    frequencies = np.fft.fftfreq(len(fft))
    
    # Find dominant frequency
    idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
    seasonal = np.real(np.fft.ifft(fft * (frequencies == frequencies[idx])))
    
    # Pad seasonal to original length
    seasonal = np.pad(seasonal, (0, len(data) - len(seasonal)), mode='wrap')
    
    # Residual
    residual = data - trend - seasonal[:len(data)]
    
    return {
        'trend': trend,
        'seasonal': seasonal[:len(data)],
        'residual': residual,
        'original': data
    }


def plot_time_series_analysis(df, save_path='time_series_analysis.png'):
    """
    Create comprehensive time series analysis plots
    
    Args:
        df: DataFrame with 'date' and 'value' columns
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Original series
    axes[0].plot(df['date'], df['value'], linewidth=1.5)
    axes[0].set_title('Original Time Series', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution
    axes[1].hist(df['value'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title('Value Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Autocorrelation
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df['value'].dropna(), ax=axes[2])
    axes[2].set_title('Autocorrelation', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Moving averages
    df['MA7'] = df['value'].rolling(window=7).mean()
    df['MA30'] = df['value'].rolling(window=30).mean()
    axes[3].plot(df['date'], df['value'], alpha=0.5, label='Original')
    axes[3].plot(df['date'], df['MA7'], label='7-day MA', linewidth=2)
    axes[3].plot(df['date'], df['MA30'], label='30-day MA', linewidth=2)
    axes[3].set_title('Moving Averages', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Date')
    axes[3].set_ylabel('Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Analysis plot saved to {save_path}")
    plt.close()


def create_sequences_multivariate(data, lookback, forecast_horizon, features):
    """
    Create sequences for multivariate time series
    
    Args:
        data: DataFrame with features
        lookback: Number of past time steps
        forecast_horizon: Number of future steps to predict
        features: List of feature column names
    
    Returns:
        X and y arrays for training
    """
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast_horizon):
        X.append(data[features].iloc[i-lookback:i].values)
        y.append(data['value'].iloc[i:i+forecast_horizon].values)
    
    return np.array(X), np.array(y)


def evaluate_forecast_accuracy(y_true, y_pred, horizons):
    """
    Evaluate forecast accuracy at different horizons
    
    Args:
        y_true: True values
        y_pred: Predicted values
        horizons: List of horizon steps to evaluate
    
    Returns:
        DataFrame with metrics for each horizon
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = []
    
    for h in horizons:
        if h <= y_true.shape[1]:
            mse = mean_squared_error(y_true[:, h-1], y_pred[:, h-1])
            mae = mean_absolute_error(y_true[:, h-1], y_pred[:, h-1])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, h-1], y_pred[:, h-1])
            
            results.append({
                'Horizon': h,
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
    
    return pd.DataFrame(results)


def save_model_info(model, history, metrics, filepath='model_info.txt'):
    """
    Save model information to text file
    
    Args:
        model: Trained Keras model
        history: Training history
        metrics: Evaluation metrics
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("="*70 + "\n\n")
        
        # Model architecture
        f.write("ARCHITECTURE:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Training history
        f.write("\n\nTRAINING HISTORY:\n")
        f.write(f"Total Epochs: {len(history.history['loss'])}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}\n")
        f.write(f"Best Validation Loss: {min(history.history['val_loss']):.6f}\n")
        
        # Metrics
        f.write("\n\nPERFORMANCE METRICS:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✅ Model info saved to {filepath}")


if __name__ == "__main__":
    print("Utility functions for time series forecasting")
    print("Import this module to use the functions in your main script")
