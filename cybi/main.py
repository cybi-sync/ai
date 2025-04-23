import argparse
import pandas as pd
import numpy as np
import logging
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

from cybi.data_module import CYBIDataProcessor
from cybi.model_architecture import CYBIModelBuilder
from cybi.training_module import CYBIModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cybi_main.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CYBISystem:
    """
    Main CYBI Smartshoe AI System that integrates all components for 
    weight prediction and health monitoring.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the CYBI system with optional config file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        # Load configuration if provided
        self.config = self._load_config(config_path)
        
        # Create core components
        self.data_processor = CYBIDataProcessor(self.config.get('data_processing'))
        self.model_builder = CYBIModelBuilder(self.config.get('model_architecture'))
        self.trainer = CYBIModelTrainer(self.config.get('training'))
        
        # Create directories
        os.makedirs('./data', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./visualizations', exist_ok=True)
        
        logger.info("CYBI AI System initialized")
    
    def _load_config(self, config_path):
        """Load configuration from JSON file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        else:
            logger.info("Using default configuration")
            return {}
    
    def generate_synthetic_data(self, n_samples=10000, n_sensors=24, output_path='./data/synthetic_data.csv'):
        """
        Generate synthetic smartshoe data for development and testing.
        
        Args:
            n_samples: Number of time samples
            n_sensors: Number of sensors in the shoe
            output_path: Path to save the generated data
            
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating synthetic data with {n_samples} samples and {n_sensors} sensors")
        
        # Time points
        time_points = np.linspace(0, 3600, n_samples)  # 1 hour of data
        
        # Create empty DataFrame
        columns = []
        data = np.zeros((n_samples, n_sensors + 2))  # +2 for weight and health
        
        # Pressure sensors (12 sensors)
        for i in range(12):
            side = 'left' if i < 6 else 'right'
            position = ['heel', 'arch', 'mid', 'ball', 'toe', 'edge'][i % 6]
            columns.append(f"pressure_{side}_{position}")
        
        # Accelerometer (3 axes x 2 shoes)
        for side in ['left', 'right']:
            for axis in ['x', 'y', 'z']:
                columns.append(f"accel_{side}_{axis}")
        
        # Gyroscope (3 axes x 2 shoes)
        for side in ['left', 'right']:
            for axis in ['x', 'y', 'z']:
                columns.append(f"gyro_{side}_{axis}")
        
        # Target columns
        columns.append('weight')
        columns.append('health_status')
        
        # Generate synthetic sensor data with walking patterns
        for i in range(n_samples):
            t = time_points[i]
            
            # Basic walking pattern with variations
            step_pattern = np.sin(t * 2)
            
            # Pressure sensors
            for j in range(12):
                # Each sensor has different phase and amplitude
                phase = j * np.pi / 6
                amplitude = 0.8 + 0.4 * np.random.random()
                
                # Left-right differences for asymmetry
                asymmetry = 0.2 if j < 6 else -0.2  # Left vs right foot
                
                # Add some randomness and variation
                data[i, j] = max(0, amplitude * (np.sin(t * 2 + phase) + asymmetry) + 0.2 * np.random.randn())
            
            # Accelerometer
            for j in range(12, 18):
                axis = (j - 12) % 3
                
                # Different patterns for different axes
                if axis == 0:  # X axis - side to side
                    data[i, j] = 0.5 * np.sin(t * 2) + 0.3 * np.random.randn()
                elif axis == 1:  # Y axis - forward motion
                    data[i, j] = np.sin(t * 2) + 0.3 * np.random.randn()
                else:  # Z axis - up down
                    data[i, j] = 0.8 * np.abs(np.sin(t * 2)) + 0.3 * np.random.randn()
            
            # Gyroscope
            for j in range(18, 24):
                axis = (j - 18) % 3
                
                # Different rotation patterns
                if axis == 0:  # Roll
                    data[i, j] = 0.4 * np.sin(t * 2) + 0.2 * np.random.randn()
                elif axis == 1:  # Pitch
                    data[i, j] = 0.6 * np.sin(t * 2 + np.pi/4) + 0.2 * np.random.randn()
                else:  # Yaw
                    data[i, j] = 0.3 * np.sin(t * 2 + np.pi/2) + 0.2 * np.random.randn()
        
        # Generate weight data (in kg) - let's make it depend on pressure patterns
        base_weight = 75  # Base weight in kg
        weight_variation = 15  # Variation range in kg
        
        # Sum all pressure values for each sample and map to weight
        pressure_sum = np.sum(data[:, 0:12], axis=1)
        normalized_pressure = (pressure_sum - np.min(pressure_sum)) / (np.max(pressure_sum) - np.min(pressure_sum))
        
        # Map to weight range with slight randomness
        weights = base_weight - weight_variation/2 + weight_variation * normalized_pressure + 2 * np.random.randn(n_samples)
        data[:, -2] = weights
        
        # Generate health status (0 = healthy, 1-14 = various conditions)
        # Let's make it depend on movement patterns
        
        # Calculate gait asymmetry (difference between left and right pressure)
        left_pressure = np.sum(data[:, 0:6], axis=1)
        right_pressure = np.sum(data[:, 6:12], axis=1)
        asymmetry = np.abs(left_pressure - right_pressure) / (left_pressure + right_pressure + 1e-6)
        
        # Calculate impact forces
        impact = np.max(np.abs(data[:, 12:18]), axis=1)
        
        # Calculate gait regularity
        regularity = np.std(data[:, 12:18], axis=1)
        
        # Calculate probability of health issues
        health_probs = np.zeros((n_samples, 15))  # 15 classes (0 = healthy, 1-14 = conditions)
        
        # Class 0: Healthy - higher probability when asymmetry is low
        health_probs[:, 0] = 1 - asymmetry
        
        # Classes 1-14: Various conditions
        # Class 1: High asymmetry (possible limping)
        health_probs[:, 1] = asymmetry
        
        # Class 2: High impact (possible joint issues)
        health_probs[:, 2] = (impact - np.min(impact)) / (np.max(impact) - np.min(impact))
        
        # Class 3: Irregular gait (possible neurological issues)
        health_probs[:, 3] = (regularity - np.min(regularity)) / (np.max(regularity) - np.min(regularity))
        
        # Other conditions: random assignment with low probability
        for j in range(4, 15):
            health_probs[:, j] = 0.1 * np.random.random(n_samples)
        
        # Normalize probabilities to sum to 1
        health_probs = health_probs / np.sum(health_probs, axis=1)[:, np.newaxis]
        
        # Sample health status based on probabilities
        health_status = np.array([np.random.choice(15, p=health_probs[i]) for i in range(n_samples)])
        data[:, -1] = health_status
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic data saved to {output_path}")
        
        return df
    
    def train_models(self, data_path, weight_target='weight', health_target='health_status'):
        """
        Train both weight prediction and health monitoring models.
        
        Args:
            data_path: Path to the input data
            weight_target: Column name for weight prediction target
            health_target: Column name for health prediction target
            
        Returns:
            Dictionary with trained models and evaluation metrics
        """
        logger.info(f"Loading data from {data_path}")
        raw_data = pd.read_csv(data_path)
        
        results = {}
        
        # Train weight prediction model
        logger.info("Starting weight prediction model training")
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
            raw_data, weight_target, task_type='regression'
        )
        
        weight_model = self.trainer.train_weight_prediction_model(X_train, X_val, y_train, y_val)
        weight_metrics = self.trainer.evaluate_weight_prediction_model(weight_model, X_test, y_test)
        
        # Save model
        weight_model.save('./models/weight_prediction_model')
        logger.info("Weight prediction model saved")
        
        # Train health prediction model
        logger.info("Starting health prediction model training")
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
            raw_data, health_target, task_type='classification'
        )
        
        health_model = self.trainer.train_health_prediction_model(X_train, X_val, y_train, y_val)
        
        # Handle ensemble models (list of models)
        if isinstance(health_model, list):
            health_metrics = self.trainer.evaluate_health_prediction_model(health_model, X_test, y_test)
            
            # Save individual models
            for i, model in enumerate(health_model):
                model.save(f'./models/health_prediction_ensemble_model_{i}')
        else:
            health_metrics = self.trainer.evaluate_health_prediction_model(health_model, X_test, y_test)
            health_model.save('./models/health_prediction_model')
        
        logger.info("Health prediction model saved")
        
        # Store results
        results['weight_metrics'] = weight_metrics
        results['health_metrics'] = health_metrics
        
        # Save metrics to file
        with open('./results/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return {
            'weight_model': weight_model,
            'health_model': health_model,
            'weight_metrics': weight_metrics,
            'health_metrics': health_metrics
        }
    
    def predict(self, data, models=None):
        """
        Make predictions using trained models.
        
        Args:
            data: Raw input data from smartshoe sensors
            models: Dictionary with 'weight_model' and 'health_model' keys,
                   or None to load from saved models
                   
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Making predictions on data with shape {data.shape}")
        
        # Process data
        processed_data = self.data_processor.process(data)
        
        # Load models if not provided
        if models is None:
            logger.info("Loading models from disk")
            try:
                weight_model = tf.keras.models.load_model('./models/weight_prediction_model')
                
                # Check if ensemble health models exist
                ensemble_model_paths = [f for f in os.listdir('./models') if f.startswith('health_prediction_ensemble_model_')]
                
                if ensemble_model_paths:
                    health_model = []
                    for path in sorted(ensemble_model_paths):
                        health_model.append(tf.keras.models.load_model(f'./models/{path}'))
                else:
                    health_model = tf.keras.models.load_model('./models/health_prediction_model')
                
                models = {
                    'weight_model': weight_model,
                    'health_model': health_model
                }
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                raise
        
        # Make predictions
        # Reshape for temporal models if needed
        input_data = processed_data.copy()
        
        # For weight prediction
        weight_input_shape = models['weight_model'].input_shape
        if len(weight_input_shape) > 2:  # If temporal (3D)
            window_size = weight_input_shape[1]
            n_features = weight_input_shape[2]
            
            # Ensure we have at least window_size samples
            if len(input_data) < window_size:
                # Pad with zeros
                padding = pd.DataFrame(
                    np.zeros((window_size - len(input_data), input_data.shape[1])),
                    columns=input_data.columns
                )
                input_data = pd.concat([input_data, padding])
            
            # Reshape to 3D
            weight_input = np.array(input_data).reshape(1, len(input_data), input_data.shape[1])
        else:  # If not temporal (2D)
            weight_input = input_data
        
        # Make weight prediction
        weight_prediction = models['weight_model'].predict(weight_input)
        
        # For health prediction
        if isinstance(models['health_model'], list):
            # Ensemble prediction
            health_predictions = []
            
            for model in models['health_model']:
                health_input_shape = model.input_shape
                
                if len(health_input_shape) > 2:  # If temporal (3D)
                    window_size = health_input_shape[1]
                    n_features = health_input_shape[2]
                    
                    # Ensure we have at least window_size samples
                    if len(input_data) < window_size:
                        # Pad with zeros
                        padding = pd.DataFrame(
                            np.zeros((window_size - len(input_data), input_data.shape[1])),
                            columns=input_data.columns
                        )
                        input_data = pd.concat([input_data, padding])
                    
                    # Reshape to 3D
                    health_input = np.array(input_data).reshape(1, len(input_data), input_data.shape[1])
                else:  # If not temporal (2D)
                    health_input = input_data
                
                health_predictions.append(model.predict(health_input))
            
            # Average ensemble predictions
            health_prediction = np.mean(health_predictions, axis=0)
        else:
            # Single model prediction
            health_input_shape = models['health_model'].input_shape
            
            if len(health_input_shape) > 2:  # If temporal (3D)
                window_size = health_input_shape[1]
                n_features = health_input_shape[2]
                
                # Ensure we have at least window_size samples
                if len(input_data) < window_size:
                    # Pad with zeros
                    padding = pd.DataFrame(
                        np.zeros((window_size - len(input_data), input_data.shape[1])),
                        columns=input_data.columns
                    )
                    input_data = pd.concat([input_data, padding])
                
                # Reshape to 3D
                health_input = np.array(input_data).reshape(1, len(input_data), input_data.shape[1])
            else:  # If not temporal (2D)
                health_input = input_data
            
            health_prediction = models['health_model'].predict(health_input)
        
        # Format results
        results = pd.DataFrame()
        results['weight_prediction'] = weight_prediction.flatten()
        
        # For classification, get class with highest probability
        if health_prediction.shape[1] > 1:
            results['health_class'] = np.argmax(health_prediction, axis=1)
            
            # Add probability for each class
            for i in range(health_prediction.shape[1]):
                results[f'health_prob_class_{i}'] = health_prediction[:, i]
        else:
            # Binary classification
            results['health_class'] = (health_prediction > 0.5).astype(int).flatten()
            results['health_prob'] = health_prediction.flatten()
        
        logger.info(f"Predictions complete: {len(results)} predictions generated")
        
        return results
    
    def run_pipeline(self, data_path=None, generate_data=False, n_samples=10000):
        """
        Run the full pipeline: data preparation, training, and evaluation.
        
        Args:
            data_path: Path to input data, or None to generate synthetic data
            generate_data: Whether to generate synthetic data
            n_samples: Number of samples for synthetic data
            
        Returns:
            Dictionary with results
        """
        # Generate or load data
        if generate_data or data_path is None:
            data_path = './data/synthetic_data.csv'
            self.generate_synthetic_data(n_samples=n_samples, output_path=data_path)
        
        # Train models
        results = self.train_models(data_path)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate a summary report of the training and evaluation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open('./results/summary_report.md', 'w') as f:
            f.write(f"# CYBI Smartshoe AI System - Summary Report\n\n")
            f.write(f"Generated on: {timestamp}\n\n")
            
            f.write("## Weight Prediction Model\n\n")
            f.write("### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in results['weight_metrics'].items():
                f.write(f"| {metric} | {value:.4f} |\n")
            
            f.write("\n## Health Prediction Model\n\n")
            f.write("### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in results['health_metrics'].items():
                if isinstance(value, float):
                    f.write(f"| {metric} | {value:.4f} |\n")
                else:
                    f.write(f"| {metric} | {value} |\n")
            
            f.write("\n## System Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2))
            f.write("\n```\n")
        
        logger.info("Summary report generated: ./results/summary_report.md")

def main():
    """Main function to run CYBI system from command line."""
    parser = argparse.ArgumentParser(description="CYBI Smartshoe AI System")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to input data')
    parser.add_argument('--generate_data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples for synthetic data')
    
    args = parser.parse_args()
    
    # Create CYBI system
    cybi_system = CYBISystem(config_path=args.config)
    
    # Run pipeline
    results = cybi_system.run_pipeline(
        data_path=args.data,
        generate_data=args.generate_data,
        n_samples=args.n_samples
    )
    
    logger.info("CYBI system pipeline complete")

if __name__ == '__main__':
    main() 