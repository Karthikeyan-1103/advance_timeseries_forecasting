# 🌟 Advanced Time Series Forecasting with Neural Networks

A comprehensive deep learning project for time series forecasting using LSTM (Long Short-Term Memory) networks and advanced neural network architectures.

## 📋 Project Overview

This project implements a complete time series forecasting solution that can:
- Generate synthetic time series data with trend, seasonality, and noise
- Preprocess and prepare data for deep learning models
- Build and train sophisticated LSTM networks
- Evaluate model performance with multiple metrics
- Visualize predictions and training history
- Forecast future values beyond the dataset

## 🎯 Features

### 1. **Data Generation**
- Synthetic time series with realistic patterns
- Trend, seasonality, and noise components
- Customizable data length and characteristics

### 2. **Model Architecture**
- **Bidirectional LSTM layers** for capturing temporal dependencies
- **Dropout layers** for regularization (20% dropout rate)
- **Dense layers** for final prediction refinement
- **Huber loss** for robustness to outliers
- **Adam optimizer** with learning rate scheduling

### 3. **Advanced Training**
- Early stopping to prevent overfitting
- Model checkpointing to save best weights
- Learning rate reduction on plateau
- Validation monitoring

### 4. **Comprehensive Evaluation**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score for goodness of fit

### 5. **Visualization**
- Training history plots
- Prediction comparisons
- Error distributions
- Multi-step forecast visualizations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Installation

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

Run the main script:
```bash
python main.py
```

This will:
1. Generate synthetic time series data
2. Prepare and split the data
3. Build the LSTM model
4. Train the model (may take 5-15 minutes depending on hardware)
5. Evaluate performance
6. Generate visualizations
7. Create future forecasts

## 📊 Project Structure

```
time-series-forecasting/
│
├── main.py                    # Main execution script
├── requirements.txt           # Project dependencies
├── README.md                  # This file
│
├── best_model.keras          # Trained model (generated after training)
└── forecasting_results.png   # Visualization output (generated)
```

## 🔧 Configuration

You can customize the forecasting parameters in `main.py`:

```python
forecaster = TimeSeriesForecaster(
    lookback=60,           # Number of past time steps to use
    forecast_horizon=10    # Number of future steps to predict
)

# Generate custom data
df = forecaster.generate_synthetic_data(n_samples=1000)

# Adjust training parameters
history = forecaster.train_model(
    epochs=100,           # Maximum training epochs
    batch_size=32         # Batch size for training
)
```

## 📈 Model Architecture

```
Layer (type)                 Output Shape              Params
================================================================
Bidirectional LSTM (128)     (None, 60, 256)          133,120
Dropout (0.2)                (None, 60, 256)          0
Bidirectional LSTM (64)      (None, 60, 128)          164,352
Dropout (0.2)                (None, 60, 128)          0
LSTM (32)                    (None, 32)               20,608
Dropout (0.2)                (None, 32)               0
Dense (64)                   (None, 64)               2,112
Dropout (0.2)                (None, 64)               0
Dense (32)                   (None, 32)               2,080
Dense (forecast_horizon)     (None, 10)               330
================================================================
Total params: 322,602
```

## 🎓 Key Concepts

### LSTM (Long Short-Term Memory)
- Specialized RNN architecture for sequential data
- Capable of learning long-term dependencies
- Addresses vanishing gradient problem

### Bidirectional LSTM
- Processes sequences in both forward and backward directions
- Captures context from past and future
- Improves prediction accuracy

### Multi-Step Forecasting
- Predicts multiple future time steps simultaneously
- More efficient than iterative single-step prediction
- Maintains temporal relationships

### Time Series Components
- **Trend**: Long-term increase or decrease
- **Seasonality**: Regular periodic patterns
- **Noise**: Random fluctuations

## 📊 Performance Metrics

The model is evaluated using:

1. **MSE (Mean Squared Error)**: Heavily penalizes large errors
2. **MAE (Mean Absolute Error)**: Average magnitude of errors
3. **RMSE (Root Mean Squared Error)**: In same units as target variable
4. **R² Score**: Proportion of variance explained (closer to 1 is better)

## 🔮 Future Forecasting

To forecast beyond your dataset:

```python
# Forecast 30 steps into the future
future_predictions = forecaster.forecast_future(n_steps=30)
```

## 💡 Usage Examples

### Load Trained Model

```python
from tensorflow import keras
import numpy as np

# Load the saved model
model = keras.models.load_model('best_model.keras')

# Prepare your input data (shape: [1, lookback, 1])
input_data = your_scaled_data[-60:].reshape(1, 60, 1)

# Make predictions
predictions = model.predict(input_data)
```

### Use with Real Data

```python
import pandas as pd

# Load your CSV file
df = pd.read_csv('your_data.csv')

# Ensure it has 'date' and 'value' columns
# df = df.rename(columns={'your_date_col': 'date', 'your_value_col': 'value'})

# Use the forecaster
forecaster = TimeSeriesForecaster()
forecaster.df = df
X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
# ... continue with training
```

## 🐛 Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution**: Install TensorFlow
```bash
pip install tensorflow>=2.15.0
```

### Issue: Training is very slow
**Solution**: 
- Reduce batch size or epochs
- Use GPU if available
- Reduce model complexity (fewer LSTM units)

### Issue: Poor predictions
**Solution**:
- Increase training data size
- Adjust lookback window
- Tune hyperparameters (learning rate, layers, units)
- Check data quality and preprocessing

## 📚 References

- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf) - Original paper
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 🤝 Contributing

To extend this project:
1. Add more data preprocessing techniques
2. Implement additional model architectures (GRU, Transformer)
3. Add hyperparameter optimization
4. Include real-world datasets
5. Deploy as web application or API

## 📄 License

This project is open source and available for educational and commercial use.

## 👨‍💻 Author

Created as a comprehensive demonstration of time series forecasting with deep learning.

## 🙏 Acknowledgments

- TensorFlow team for the excellent framework
- scikit-learn for preprocessing utilities
- The deep learning community for research and insights

---

## 🚦 Quick Reference

### Installation
```bash
pip install -r requirements.txt
```

### Run Project
```bash
python main.py
```

### Check GPU
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Adjust Model Size
For faster training on CPU:
```python
# In build_lstm_model(), reduce units:
LSTM(64, ...)  # Instead of 128
LSTM(32, ...)  # Instead of 64
```

---

**Happy Forecasting! 📈🔮**
