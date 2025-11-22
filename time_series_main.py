"""
Advanced Time Series Forecasting with Neural Networks
Main execution script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TimeSeriesForecaster:
    """
    Advanced Time Series Forecasting using LSTM and Neural Networks
    """
    
    def __init__(self, lookback=60, forecast_horizon=10):
        """
        Initialize the forecaster
        
        Args:
            lookback: Number of past time steps to use for prediction
            forecast_horizon: Number of future time steps to predict
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic time series data with trend, seasonality, and noise
        """
        print("📊 Generating synthetic time series data...")
        
        # Time array
        t = np.arange(n_samples)
        
        # Components
        trend = 0.05 * t
        seasonality = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 25)
        noise = np.random.normal(0, 2, n_samples)
        
        # Combine components
        series = trend + seasonality + noise + 50
        
        # Create DataFrame
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        self.df = pd.DataFrame({'date': dates, 'value': series})
        
        print(f"✅ Generated {n_samples} data points")
        return self.df
    
    def prepare_data(self, data, train_split=0.8):
        """
        Prepare data for training: scaling and creating sequences
        """
        print("\n🔧 Preparing data...")
        
        # Extract values
        values = data['value'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data) - self.forecast_horizon):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i:i+self.forecast_horizon, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split into train and test
        split_idx = int(len(X) * train_split)
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Training samples: {len(self.X_train)}")
        print(f"✅ Testing samples: {len(self.X_test)}")
        print(f"✅ Input shape: {self.X_train.shape}")
        print(f"✅ Output shape: {self.y_train.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_lstm_model(self):
        """
        Build advanced LSTM model for time series forecasting
        """
        print("\n🏗️ Building LSTM model...")
        
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(128, return_sequences=True, 
                              input_shape=(self.lookback, 1))),
            Dropout(0.2),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for forecasting
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("✅ Model built successfully")
        print(f"\n{model.summary()}")
        
        return model
    
    def train_model(self, epochs=100, batch_size=32):
        """
        Train the LSTM model with callbacks
        """
        print("\n🚀 Training model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate model performance with multiple metrics
        """
        print("\n📈 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Inverse transform predictions and actual values
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        y_test_inv = self.scaler.inverse_transform(self.y_test)
        
        # Calculate metrics for each forecast horizon
        metrics = {}
        for i in range(self.forecast_horizon):
            mse = mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i])
            mae = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_inv[:, i], y_pred_inv[:, i])
            
            metrics[f'step_{i+1}'] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
        
        # Overall metrics
        overall_mse = mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
        overall_mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
        
        print("\n📊 Overall Performance Metrics:")
        print(f"  MSE: {overall_mse:.4f}")
        print(f"  MAE: {overall_mae:.4f}")
        print(f"  RMSE: {overall_rmse:.4f}")
        print(f"  R² Score: {overall_r2:.4f}")
        
        return metrics, y_pred_inv, y_test_inv
    
    def plot_results(self, y_pred, y_test):
        """
        Visualize training history and predictions
        """
        print("\n📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Training History - Loss
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Training History - MAE
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        plt.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        plt.title('Mean Absolute Error Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. First prediction example
        ax3 = plt.subplot(2, 3, 3)
        sample_idx = 0
        plt.plot(range(self.forecast_horizon), y_test[sample_idx], 
                marker='o', label='Actual', linewidth=2)
        plt.plot(range(self.forecast_horizon), y_pred[sample_idx], 
                marker='s', label='Predicted', linewidth=2)
        plt.title('Sample Forecast (First Test Sample)', fontsize=14, fontweight='bold')
        plt.xlabel('Forecast Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Multiple predictions
        ax4 = plt.subplot(2, 3, 4)
        for i in range(min(5, len(y_test))):
            plt.plot(range(self.forecast_horizon), y_test[i], 
                    alpha=0.5, linewidth=1, color='blue')
            plt.plot(range(self.forecast_horizon), y_pred[i], 
                    alpha=0.5, linewidth=1, color='red', linestyle='--')
        plt.plot([], [], color='blue', label='Actual', linewidth=2)
        plt.plot([], [], color='red', linestyle='--', label='Predicted', linewidth=2)
        plt.title('Multiple Forecast Comparisons', fontsize=14, fontweight='bold')
        plt.xlabel('Forecast Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Prediction errors
        ax5 = plt.subplot(2, 3, 5)
        errors = y_test - y_pred
        plt.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 6. Time series comparison
        ax6 = plt.subplot(2, 3, 6)
        # Plot first step predictions across all test samples
        plt.plot(y_test[:, 0], label='Actual', linewidth=2, alpha=0.7)
        plt.plot(y_pred[:, 0], label='Predicted', linewidth=2, alpha=0.7)
        plt.title('First Step Predictions Across Test Set', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('forecasting_results.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'forecasting_results.png'")
        plt.show()
    
    def forecast_future(self, n_steps=30):
        """
        Forecast future values beyond the dataset
        """
        print(f"\n🔮 Forecasting {n_steps} steps into the future...")
        
        # Get last sequence from data
        last_sequence = self.scaler.transform(
            self.df['value'].values[-self.lookback:].reshape(-1, 1)
        )
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps // self.forecast_horizon + 1):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, self.lookback, 1)
            
            # Predict next steps
            next_prediction = self.model.predict(current_batch, verbose=0)
            future_predictions.append(next_prediction[0])
            
            # Update sequence with first predicted value
            current_sequence = np.vstack([
                current_sequence[1:],
                next_prediction[0, 0].reshape(1, 1)
            ])
        
        # Flatten and trim predictions
        future_predictions = np.concatenate(future_predictions)[:n_steps]
        
        # Inverse transform
        future_predictions = self.scaler.inverse_transform(
            future_predictions.reshape(-1, 1)
        ).flatten()
        
        print("✅ Future forecast completed")
        return future_predictions


def main():
    """
    Main execution function
    """
    print("="*70)
    print("🌟 Advanced Time Series Forecasting with Neural Networks 🌟")
    print("="*70)
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster(lookback=60, forecast_horizon=10)
    
    # Generate data
    df = forecaster.generate_synthetic_data(n_samples=1000)
    
    # Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
    
    # Build model
    model = forecaster.build_lstm_model()
    
    # Train model
    history = forecaster.train_model(epochs=100, batch_size=32)
    
    # Evaluate model
    metrics, y_pred, y_test_inv = forecaster.evaluate_model()
    
    # Plot results
    forecaster.plot_results(y_pred, y_test_inv)
    
    # Forecast future
    future_forecast = forecaster.forecast_future(n_steps=30)
    
    print("\n" + "="*70)
    print("✅ Project completed successfully!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  - best_model.keras (trained model)")
    print("  - forecasting_results.png (visualizations)")
    print("\n🎯 Next steps:")
    print("  - Load model: model = keras.models.load_model('best_model.keras')")
    print("  - Make predictions on new data")
    print("  - Fine-tune hyperparameters")
    print("  - Deploy model to production")


if __name__ == "__main__":
    main()
