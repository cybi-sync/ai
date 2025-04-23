# CYBI Smartshoe AI System

## Overview
The CYBI Smartshoe AI System is a comprehensive solution for predicting human weight and providing early health condition warnings based on data collected from CYBI smart shoes. This system employs advanced machine learning and deep learning techniques to analyze sensor data and make accurate predictions.

## Key Features
- **Weight Prediction**: Accurately predicts the user's weight based on pressure distribution patterns
- **Health Monitoring**: Detects potential health issues through gait analysis and movement patterns
- **Real-time Analysis**: Processes sensor data in real-time for immediate feedback
- **Multi-modal Learning**: Combines data from pressure sensors, accelerometers, and gyroscopes
- **Advanced Neural Architectures**: Implements state-of-the-art deep learning architectures including CNNs, RNNs, and Transformers

## Architecture
The system consists of several key components:

1. **Data Processing Module**
   - Advanced signal processing and feature extraction
   - Sensor fusion from multiple data sources
   - Anomaly detection and filtering
   - Wavelet transformations and frequency domain analysis

2. **Model Architecture**
   - Hybrid CNN-LSTM networks for temporal and spatial patterns
   - Transformer-based architectures for capturing long-range dependencies
   - Residual connections and attention mechanisms
   - Multi-stream architecture for health detection

3. **Training Module**
   - Sophisticated training strategies with regularization
   - Ensemble methods for improved robustness
   - Cross-validation and hyperparameter optimization
   - Imbalanced data handling

4. **Deployment API**
   - FastAPI-based REST interface
   - Real-time prediction endpoints
   - Batch processing capabilities
   - Background processing for large datasets

## Technical Stack
- **Programming Language**: Python 3.8+
- **Deep Learning**: TensorFlow 2.10+, Keras
- **Machine Learning**: scikit-learn, XGBoost, CatBoost, LightGBM
- **Signal Processing**: SciPy, PyWavelets, tslearn
- **Data Handling**: NumPy, Pandas
- **Deployment**: FastAPI, Uvicorn
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: SHAP, LIME

## System Requirements
- Python 3.8 or higher
- 8GB+ RAM
- CUDA-compatible GPU recommended for training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cybi-smartshoe-ai.git
cd cybi-smartshoe-ai

# Install dependencies
pip install -r requirements.txt

# Set up directories
mkdir -p data models results logs visualizations
```

## Quick Start

### Generate Synthetic Data
```bash
python -m cybi.main --generate_data --n_samples 10000
```

### Train Models
```bash
python -m cybi.main --data ./data/synthetic_data.csv
```

### Start API Server
```bash
python -m cybi.deploy_api
```

### Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @./examples/sample_data.json
```

## Model Details

### Weight Prediction Model
- **Architecture**: Hybrid CNN-LSTM with residual connections
- **Input**: 100 time steps, 50 features per step
- **Performance**: RMSE < 2kg on test data

### Health Monitoring Model
- **Architecture**: Multi-stream network with CNN, LSTM, and Transformer pathways
- **Input**: 200 time steps, 75 features per step
- **Classes**: 15 different health conditions and normal state
- **Performance**: >90% accuracy on test data

## Project Structure
```
cybi/
  ├── data_module.py        # Data processing and feature extraction
  ├── model_architecture.py # Neural network architectures
  ├── training_module.py    # Training pipelines and evaluation
  ├── main.py               # Main system integration
  ├── deploy_api.py         # REST API for deployment
  └── utils.py              # Helper functions and utilities
data/                       # Data storage directory
models/                     # Saved model weights and configurations
results/                    # Evaluation metrics and predictions
visualizations/             # Plots and visualizations
```

## Configuration
The system is highly configurable through JSON configuration files. Example:

```json
{
  "data_processing": {
    "scaling_method": "robust",
    "anomaly_detection": "iforest",
    "feature_selection_method": "mutual_info"
  },
  "model_architecture": {
    "weight_prediction": {
      "architecture": "hybrid",
      "lstm_units": [64, 128],
      "cnn_filters": [64, 128, 256]
    }
  },
  "training": {
    "batch_size": 32,
    "epochs": 150,
    "early_stopping": {
      "enabled": true,
      "patience": 20
    }
  }
}
```

## API Documentation
Once the API server is running, visit http://localhost:8000/docs for the interactive API documentation.

## Example Use Cases
1. **Personal Health Monitoring**: Daily weight tracking and early detection of gait abnormalities
2. **Clinical Applications**: Monitoring patients' recovery progress after injury or surgery
3. **Sports Performance**: Tracking athletic performance and detecting potential injury risks
4. **Elderly Care**: Monitoring for fall risks and early signs of mobility issues

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- The CYBI Smartshoe hardware team
- TensorFlow and Keras development teams
- scikit-learn community for machine learning tools
- FastAPI developers for the excellent API framework 