# 🚀 Complete Setup Guide - Time Series Forecasting Project

This guide will walk you through setting up and running the Time Series Forecasting project from scratch.

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Step-by-Step Installation](#step-by-step-installation)
3. [Project Structure](#project-structure)
4. [Running the Project](#running-the-project)
5. [Customization Guide](#customization-guide)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: Any modern processor

### Recommended for Faster Training
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **CUDA**: Version 11.2 or higher
- **cuDNN**: Version 8.1 or higher

---

## 📦 Step-by-Step Installation

### Step 1: Install Python

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"**
3. Verify installation:
```bash
python --version
```

#### macOS
```bash
# Using Homebrew
brew install python@3.11
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3-pip python3-venv
```

### Step 2: Create Project Directory

```bash
# Create project folder
mkdir time-series-forecasting
cd time-series-forecasting
```

### Step 3: Set Up Virtual Environment

#### Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt when activated.

### Step 4: Create Project Files

Create the following files in your project directory:

1. **main.py** - Main execution script (copy content from artifact)
2. **utils.py** - Utility functions (copy content from artifact)
3. **config.py** - Configuration settings (copy content from artifact)
4. **requirements.txt** - Dependencies (copy content from artifact)
5. **README.md** - Project documentation (copy content from artifact)

### Step 5: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- NumPy (numerical computing)
- Pandas (data manipulation)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (preprocessing)
- TensorFlow (deep learning)

**Installation may take 5-10 minutes depending on your internet speed.**

### Step 6: Verify Installation

```bash
# Test imports
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import numpy; print('NumPy installed successfully')"
python -c "import pandas; print('Pandas installed successfully')"
```

---

## 📁 Project Structure

After setup, your directory should look like this:

```
time-series-forecasting/
│
├── venv/                      # Virtual environment (don't commit to git)
│
├── main.py                    # Main execution script
├── utils.py                   # Utility functions
├── config.py                  # Configuration settings
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── SETUP_GUIDE.md            # This file
│
├── data/                      # (Created automatically)
│   └── your_data.csv
│
├── models/                    # (Created after training)
│   ├── best_model.keras
│   └── scaler.pkl
│
├── results/                   # (Created after running)
│   ├── forecasting_results.png
│   ├── time_series_analysis.png
│   └── metrics.json
│
└── logs/                      # (Optional, for logging)
    └── training.log
```

---

## 🎯 Running the Project

### Quick Start (5 minutes)

1. **Activate virtual environment** (if not already activated):
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. **Run the main script**:
```bash
python main.py
```

3. **What happens**:
   - Generates synthetic data (1000 samples)
   - Splits into train/test sets (80/20)
   - Builds LSTM model
   - Trains for up to 100 epochs (early stopping enabled)
   - Evaluates performance
   - Creates visualizations
   - Saves trained model

4. **Expected output**:
```
======================================================================
🌟 Advanced Time Series Forecasting with Neural Networks 🌟
======================================================================
📊 Generating synthetic time series data...
✅ Generated 1000 data points

🔧 Preparing data...
✅ Training samples: 720
✅ Testing samples: 220

🏗️ Building LSTM model...
✅ Model built successfully

🚀 Training model...
[Training progress with epochs]
✅ Training completed!

📈 Evaluating model...
📊 Overall Performance Metrics:
  MSE: 2.3456
  MAE: 1.2345
  RMSE: 1.5318
  R² Score: 0.9542

✅ Project completed successfully!
```

### Training Time Estimates

- **CPU only**: 10-20 minutes
- **GPU (CUDA enabled)**: 2-5 minutes

---

## ⚙️ Customization Guide

### 1. Adjust Model Parameters

Edit `main.py`:

```python
# Change lookback window and forecast horizon
forecaster = TimeSeriesForecaster(
    lookback=90,           # Increased from 60
    forecast_horizon=20    # Increased from 10
)
```

### 2. Modify Training Settings

```python
# Adjust epochs and batch size
history = forecaster.train_model(
    epochs=50,      # Reduced for faster training
    batch_size=64   # Larger batch size
)
```

### 3. Use Your Own Data

```python
import pandas as pd

# Load your CSV file
df = pd.read_csv('your_data.csv')

# Ensure columns are named 'date' and 'value'
df = df.rename(columns={
    'timestamp': 'date',
    'price': 'value'
})

# Use with forecaster
forecaster.df = df
X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
```

### 4. Change Model Architecture

Edit `build_lstm_model()` in `main.py`:

```python
def build_lstm_model(self):
    model = Sequential([
        # Adjust number of units here
        Bidirectional(LSTM(256, return_sequences=True, 
                          input_shape=(self.lookback, 1))),
        Dropout(0.3),
        
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dense(self.forecast_horizon)
    ])
    return model
```

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**:
```bash
pip install tensorflow>=2.15.0
```

### Problem: Training is extremely slow

**Solutions**:
1. Reduce model size:
```python
# Use smaller LSTM units
LSTM(64, ...)  # Instead of 128
```

2. Reduce epochs:
```python
forecaster.train_model(epochs=20)  # Instead of 100
```

3. Use GPU (if available)

### Problem: "Out of memory" error

**Solutions**:
1. Reduce batch size:
```python
forecaster.train_model(batch_size=16)  # Instead of 32
```

2. Reduce lookback window:
```python
TimeSeriesForecaster(lookback=30)  # Instead of 60
```

### Problem: Poor prediction accuracy

**Solutions**:
1. Increase training data size
2. Adjust lookback window
3. Add more training epochs
4. Try different architectures
5. Tune hyperparameters

### Problem: GPU not detected

**Check GPU availability**:
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

**Install GPU support**:
```bash
# For NVIDIA GPUs
pip install tensorflow[and-cuda]
```

---

## 🎓 Advanced Usage

### 1. Using Configuration File

```python
from config import MODEL_CONFIG, TRAINING_CONFIG

forecaster = TimeSeriesForecaster(
    lookback=MODEL_CONFIG['lookback'],
    forecast_horizon=MODEL_CONFIG['forecast_horizon']
)

forecaster.train_model(
    epochs=TRAINING_CONFIG['epochs'],
    batch_size=TRAINING_CONFIG['batch_size']
)
```

### 2. Using Utility Functions

```python
from utils import load_csv_data, add_time_features

# Load and preprocess data
df = load_csv_data('data.csv')
df = add_time_features(df)
```

### 3. Batch Processing Multiple Datasets

```python
import glob

# Process all CSV files in directory
for filepath in glob.glob('data/*.csv'):
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath)
    forecaster = TimeSeriesForecaster()
    forecaster.df = df
    # ... continue training
```

### 4. Load Trained Model

```python
from tensorflow import keras

# Load saved model
model = keras.models.load_model('best_model.keras')

# Make predictions
predictions = model.predict(your_input_data)
```

### 5. Export Predictions

```python
import pandas as pd

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'date': test_dates,
    'actual': y_test[:, 0],
    'predicted': y_pred[:, 0]
})

# Save to CSV
predictions_df.to_csv('predictions.csv', index=False)
```

---

## 📊 Performance Optimization Tips

### For CPU Training
1. Use smaller batch sizes (16-32)
2. Reduce model complexity
3. Enable mixed precision training
4. Use fewer LSTM units

### For GPU Training
1. Use larger batch sizes (64-128)
2. Enable GPU memory growth
3. Use data generators for large datasets
4. Monitor GPU utilization

### Code for GPU Memory Growth
```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 🔐 Best Practices

1. **Always use virtual environments** to avoid dependency conflicts
2. **Version control** - Use git to track changes
3. **Save configurations** before experiments
4. **Document changes** in a log file
5. **Backup trained models** regularly
6. **Monitor training** with TensorBoard
7. **Validate on separate test set** before deployment

---

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Time Series Forecasting Guide](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [LSTM Networks Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

---

## 🆘 Getting Help

If you encounter issues:

1. Check error messages carefully
2. Review this troubleshooting section
3. Check TensorFlow/Python versions
4. Verify all dependencies are installed
5. Try with default configuration first

---

## ✅ Checklist

Before running the project, ensure:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All requirements installed (`pip install -r requirements.txt`)
- [ ] All project files in correct directory
- [ ] Sufficient disk space (2GB+)
- [ ] Dependencies verified with import tests

---

**You're all set! Run `python main.py` to start forecasting! 🚀**
