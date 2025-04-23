import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
import logging
from tqdm import tqdm
import optuna
from optuna.integration import TFKerasPruningCallback
import shap
import lime
import lime.lime_tabular
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from tensorflow.keras.models import load_model
from datetime import datetime
import json

from cybi.model_architecture import CYBIModelBuilder
from cybi.data_module import CYBIDataProcessor

logger = logging.getLogger(__name__)

class CYBIModelTrainer:
    """
    Advanced model trainer for CYBI smartshoe data, implementing various
    training strategies and validation techniques.
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'data': {
                'train_test_split': 0.2,
                'val_split': 0.15,
                'random_state': 42,
                'shuffle': True,
                'stratify': True,
                'cross_validation': {
                    'enabled': True,
                    'n_splits': 5,
                    'shuffle': True
                },
                'class_balancing': {
                    'enabled': True,
                    'method': 'smote',  # Options: 'smote', 'adasyn', 'smote_tomek', 'class_weight'
                    'sampling_strategy': 'auto'
                },
                'windows': {
                    'size': 200,
                    'stride': 100,
                    'padding': 'same'
                }
            },
            'training': {
                'batch_size': 32,
                'epochs': 150,
                'early_stopping': {
                    'enabled': True,
                    'patience': 20,
                    'restore_best_weights': True,
                    'monitor': 'val_loss'
                },
                'reduce_lr': {
                    'enabled': True,
                    'patience': 10,
                    'factor': 0.5,
                    'min_lr': 1e-6,
                    'monitor': 'val_loss'
                },
                'mixed_precision': True,
                'multi_gpu': False,
                'distributed': False
            },
            'evaluation': {
                'metrics': ['mae', 'mse', 'rmse', 'r2', 'accuracy', 'precision', 'recall', 'f1', 'auc'],
                'visualization': {
                    'confusion_matrix': True,
                    'roc_curve': True,
                    'learning_curves': True,
                    'feature_importance': True,
                    'prediction_plots': True,
                    'attention_maps': True
                },
                'explanation': {
                    'shap': True,
                    'lime': True
                },
                'bootstrap': {
                    'enabled': False,
                    'n_iterations': 100,
                    'sample_size': 0.8,
                    'confidence_interval': 0.95
                }
            },
            'hyperparameter_tuning': {
                'enabled': False,
                'method': 'optuna',  # Options: 'optuna', 'grid_search', 'random_search'
                'n_trials': 50,
                'timeout': 7200,  # In seconds
                'pruning': True,
                'pruning_patience': 10
            },
            'ensemble': {
                'enabled': True,
                'method': 'stacking',  # Options: 'voting', 'stacking', 'bagging', 'boosting'
                'models': ['hybrid', 'transformer', 'cnn'],
                'weights': [0.5, 0.3, 0.2],
                'stacking_meta_model': 'xgboost'  # Options: 'xgboost', 'lightgbm', 'catboost', 'mlp'
            },
            'export': {
                'format': 'all',  # Options: 'saved_model', 'h5', 'tflite', 'onnx', 'all'
                'quantization': {
                    'enabled': True,
                    'method': 'float16'  # Options: 'float16', 'dynamic', 'full_integer'
                },
                'optimization': {
                    'enabled': True,
                    'pruning': True,
                    'clustering': False
                }
            },
            'logging': {
                'level': 'INFO',
                'save_path': './logs',
                'tensorboard': True,
                'experiment_tracking': {
                    'enabled': False,
                    'service': 'mlflow'  # Options: 'mlflow', 'wandb', 'neptune'
                }
            }
        }
        
        # Initialize the model builder
        self.model_builder = CYBIModelBuilder()
        
        # Initialize the data processor
        self.data_processor = CYBIDataProcessor()
        
        # Set up logging
        self._setup_logging()
        
        # Set up mixed precision if enabled
        if self.config['training']['mixed_precision']:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled with policy: mixed_float16")
        
        # Create directories
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./visualizations', exist_ok=True)
        os.makedirs('./explanations', exist_ok=True)
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, self.config['logging']['level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config['logging']['save_path']}/cybi_training.log"),
                logging.StreamHandler()
            ]
        )
    
    def prepare_data(self, raw_data, target_column, task_type='regression'):
        """
        Prepare and preprocess data for model training.
        
        Args:
            raw_data: Raw input data from the smartshoe sensors
            target_column: Column name for the prediction target
            task_type: 'regression' for weight prediction, 'classification' for health prediction
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"Preparing data for {task_type} task with target: {target_column}")
        
        # Process raw data using the data processor
        processed_data = self.data_processor.process(raw_data)
        
        # Handle NaN values that might have been introduced
        processed_data.fillna(0, inplace=True)
        
        # Extract features and target
        X = processed_data.drop(columns=[target_column])
        y = processed_data[target_column]
        
        # Handle classification targets
        if task_type == 'classification':
            # Encode categorical target
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            
            # Save encoder for later use
            joblib.dump(label_encoder, './models/label_encoder.pkl')
            
            # Convert to one-hot encoding for multi-class
            if len(label_encoder.classes_) > 2:
                y = to_categorical(y)
        
        # Split data
        stratify = y if self.config['data']['stratify'] and task_type == 'classification' else None
        
        # First split: train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['train_test_split'],
            random_state=self.config['data']['random_state'],
            shuffle=self.config['data']['shuffle'],
            stratify=stratify
        )
        
        # Second split: train and validation
        stratify = y_train_val if self.config['data']['stratify'] and task_type == 'classification' else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config['data']['val_split'],
            random_state=self.config['data']['random_state'],
            shuffle=self.config['data']['shuffle'],
            stratify=stratify
        )
        
        # Apply class balancing if enabled for classification
        if task_type == 'classification' and self.config['data']['class_balancing']['enabled']:
            X_train, y_train = self._apply_class_balancing(X_train, y_train)
        
        # Reshape data for temporal models if needed
        X_train, X_val, X_test = self._reshape_for_temporal_models(X_train, X_val, X_test)
        
        logger.info(f"Data preparation complete. Training set: {X_train.shape}, "
                   f"Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _apply_class_balancing(self, X, y):
        """Apply class balancing techniques for imbalanced data."""
        method = self.config['data']['class_balancing']['method']
        strategy = self.config['data']['class_balancing']['sampling_strategy']
        
        logger.info(f"Applying class balancing using {method} method")
        
        # Handle one-hot encoded targets
        if len(y.shape) > 1:
            # Convert back to 1D for resampling
            y_temp = np.argmax(y, axis=1)
        else:
            y_temp = y
        
        # Apply resampling based on configuration
        if method == 'smote':
            resampler = SMOTE(sampling_strategy=strategy, random_state=self.config['data']['random_state'])
        elif method == 'adasyn':
            resampler = ADASYN(sampling_strategy=strategy, random_state=self.config['data']['random_state'])
        elif method == 'smote_tomek':
            resampler = SMOTETomek(sampling_strategy=strategy, random_state=self.config['data']['random_state'])
        else:
            logger.warning(f"Unknown class balancing method: {method}, returning original data")
            return X, y
        
        # Apply resampling
        X_resampled, y_resampled = resampler.fit_resample(X, y_temp)
        
        # Convert back to one-hot if needed
        if len(y.shape) > 1:
            y_resampled = to_categorical(y_resampled, num_classes=y.shape[1])
        
        logger.info(f"Class distribution after balancing: {np.unique(y_temp if len(y.shape) == 1 else np.argmax(y, axis=1), return_counts=True)}")
        logger.info(f"Data shape after balancing: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def _reshape_for_temporal_models(self, X_train, X_val, X_test):
        """Reshape data for temporal models if needed."""
        # If data is already 3D (temporal), return as is
        if len(X_train.shape) == 3:
            return X_train, X_val, X_test
        
        # Get window parameters
        window_size = self.config['data']['windows']['size']
        stride = self.config['data']['windows']['stride']
        
        # Function to create sliding windows
        def create_windows(data, window_size, stride):
            n_features = data.shape[1]
            n_windows = (data.shape[0] - window_size) // stride + 1
            
            if n_windows <= 0:
                # If data is smaller than window size, pad with zeros
                padding = np.zeros((window_size - data.shape[0], n_features))
                data = np.vstack([data, padding])
                n_windows = 1
            
            windowed_data = np.zeros((n_windows, window_size, n_features))
            
            for i in range(n_windows):
                start = i * stride
                end = start + window_size
                
                if end <= data.shape[0]:
                    window = data[start:end]
                else:
                    # For last window that might be smaller than window_size
                    window = np.zeros((window_size, n_features))
                    window[:data.shape[0]-start] = data[start:]
                
                windowed_data[i] = window
            
            return windowed_data
        
        # Apply windowing to each dataset
        X_train_windowed = create_windows(X_train.values, window_size, stride)
        X_val_windowed = create_windows(X_val.values, window_size, stride)
        X_test_windowed = create_windows(X_test.values, window_size, stride)
        
        logger.info(f"Data reshaped for temporal models. New shapes - Training: {X_train_windowed.shape}, "
                   f"Validation: {X_val_windowed.shape}, Test: {X_test_windowed.shape}")
        
        return X_train_windowed, X_val_windowed, X_test_windowed
    
    def train_weight_prediction_model(self, X_train, X_val, y_train, y_val):
        """
        Train a model for weight prediction from smartshoe data.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target weights
            y_val: Validation target weights
            
        Returns:
            Trained model
        """
        logger.info("Training weight prediction model")
        
        # Build the weight prediction model
        model = self.model_builder.build_weight_prediction_model()
        
        # Prepare callbacks
        callbacks = self._get_callbacks('weight_prediction')
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self._save_training_history(history, 'weight_prediction')
        
        # Plot learning curves
        if self.config['evaluation']['visualization']['learning_curves']:
            self._plot_learning_curves(history, 'weight_prediction')
        
        logger.info("Weight prediction model training complete")
        
        return model
    
    def train_health_prediction_model(self, X_train, X_val, y_train, y_val):
        """
        Train a model for health condition prediction from smartshoe data.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target health conditions
            y_val: Validation target health conditions
            
        Returns:
            Trained model or list of models for ensemble
        """
        logger.info("Training health prediction model")
        
        # Check if ensemble is enabled
        if self.config['ensemble']['enabled'] and self.model_builder.config['health_prediction']['architecture'] == 'ensemble':
            # Train ensemble models
            return self._train_ensemble_models(X_train, X_val, y_train, y_val, task='health_prediction')
        
        # Build the health prediction model
        model = self.model_builder.build_health_prediction_model()
        
        # Prepare callbacks
        callbacks = self._get_callbacks('health_prediction')
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self._save_training_history(history, 'health_prediction')
        
        # Plot learning curves
        if self.config['evaluation']['visualization']['learning_curves']:
            self._plot_learning_curves(history, 'health_prediction')
        
        logger.info("Health prediction model training complete")
        
        return model
    
    def _train_ensemble_models(self, X_train, X_val, y_train, y_val, task='health_prediction'):
        """Train multiple models for ensemble prediction."""
        logger.info(f"Training ensemble models for {task}")
        
        # For ensemble with architecture='ensemble', we get a list of models
        models = self.model_builder.build_health_prediction_model()
        trained_models = []
        
        for i, model in enumerate(models):
            logger.info(f"Training ensemble model {i+1}/{len(models)}")
            
            # Prepare callbacks specific to this model
            callbacks = self._get_callbacks(f'{task}_ensemble_{i}')
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.config['training']['batch_size'],
                epochs=self.config['training']['epochs'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Save training history
            self._save_training_history(history, f'{task}_ensemble_{i}')
            
            # Plot learning curves
            if self.config['evaluation']['visualization']['learning_curves']:
                self._plot_learning_curves(history, f'{task}_ensemble_{i}')
            
            trained_models.append(model)
        
        logger.info(f"Ensemble training complete for {task}")
        
        return trained_models
    
    def _get_callbacks(self, model_name):
        """Get callbacks for model training."""
        callbacks = []
        
        # Model checkpoint
        callbacks.append(
            ModelCheckpoint(
                f"./models/{model_name}_best.h5",
                monitor=self.config['training']['early_stopping']['monitor'],
                save_best_only=True,
                verbose=1
            )
        )
        
        # Early stopping
        if self.config['training']['early_stopping']['enabled']:
            callbacks.append(
                EarlyStopping(
                    monitor=self.config['training']['early_stopping']['monitor'],
                    patience=self.config['training']['early_stopping']['patience'],
                    restore_best_weights=self.config['training']['early_stopping']['restore_best_weights'],
                    verbose=1
                )
            )
        
        # Learning rate reduction
        if self.config['training']['reduce_lr']['enabled']:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=self.config['training']['reduce_lr']['monitor'],
                    factor=self.config['training']['reduce_lr']['factor'],
                    patience=self.config['training']['reduce_lr']['patience'],
                    min_lr=self.config['training']['reduce_lr']['min_lr'],
                    verbose=1
                )
            )
        
        # TensorBoard
        if self.config['logging']['tensorboard']:
            log_dir = f"{self.config['logging']['save_path']}/tensorboard/{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch'
                )
            )
        
        return callbacks
    
    def _save_training_history(self, history, model_name):
        """Save training history for analysis."""
        # Convert history object to dict if needed
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        # Save as JSON
        with open(f"./results/{model_name}_history.json", 'w') as f:
            json.dump(history_dict, f)
        
        logger.info(f"Training history saved for {model_name}")
    
    def _plot_learning_curves(self, history, model_name):
        """Plot and save learning curves from training history."""
        # Convert history object to dict if needed
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        # Create figure with multiple subplots
        n_metrics = len(history_dict) // 2  # Half are training, half validation
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        
        # If only one metric, axes is not a list
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        i = 0
        for metric in history_dict:
            if not metric.startswith('val_'):
                ax = axes[i]
                
                # Plot training metric
                ax.plot(history_dict[metric], label=f'Training {metric}')
                
                # Plot validation metric if available
                val_metric = f'val_{metric}'
                if val_metric in history_dict:
                    ax.plot(history_dict[val_metric], label=f'Validation {metric}')
                
                # Add labels and legend
                ax.set_title(f'{metric} vs. Epochs')
                ax.set_xlabel('Epochs')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True)
                
                i += 1
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"./visualizations/{model_name}_learning_curves.png")
        plt.close()
        
        logger.info(f"Learning curves plotted and saved for {model_name}")
    
    def evaluate_weight_prediction_model(self, model, X_test, y_test):
        """
        Evaluate the weight prediction model and generate reports.
        
        Args:
            model: Trained weight prediction model
            X_test: Test features
            y_test: Test target weights
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating weight prediction model")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Save metrics
        with open(f"./results/weight_prediction_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Generate visualizations
        if self.config['evaluation']['visualization']['prediction_plots']:
            self._plot_regression_predictions(y_test, y_pred, 'weight_prediction')
        
        # Generate model explanations
        if self.config['evaluation']['explanation']['shap']:
            self._generate_shap_explanations(model, X_test, 'weight_prediction')
        
        if self.config['evaluation']['explanation']['lime']:
            self._generate_lime_explanations(model, X_test, y_test, 'weight_prediction')
        
        logger.info(f"Weight prediction model evaluation complete. R2 Score: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def evaluate_health_prediction_model(self, model, X_test, y_test):
        """
        Evaluate the health prediction model and generate reports.
        
        Args:
            model: Trained health prediction model or list of models for ensemble
            X_test: Test features
            y_test: Test target health conditions
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating health prediction model")
        
        # Handle ensemble models
        if isinstance(model, list):
            y_pred_proba = self._get_ensemble_predictions(model, X_test)
            
            # Convert to class predictions for metrics
            if y_pred_proba.shape[1] > 1:  # Multi-class
                y_pred = np.argmax(y_pred_proba, axis=1)
                if len(y_test.shape) > 1:  # One-hot encoded
                    y_test_classes = np.argmax(y_test, axis=1)
                else:
                    y_test_classes = y_test
            else:  # Binary
                y_pred = (y_pred_proba > 0.5).astype(int)
                y_test_classes = y_test
        else:
            # Make predictions
            y_pred_proba = model.predict(X_test)
            
            # Convert to class predictions for metrics
            if y_pred_proba.shape[1] > 1:  # Multi-class
                y_pred = np.argmax(y_pred_proba, axis=1)
                if len(y_test.shape) > 1:  # One-hot encoded
                    y_test_classes = np.argmax(y_test, axis=1)
                else:
                    y_test_classes = y_test
            else:  # Binary
                y_pred = (y_pred_proba > 0.5).astype(int)
                y_test_classes = y_test
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(y_test_classes, y_pred, y_pred_proba)
        
        # Save metrics
        with open(f"./results/health_prediction_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Generate confusion matrix
        if self.config['evaluation']['visualization']['confusion_matrix']:
            self._plot_confusion_matrix(y_test_classes, y_pred, 'health_prediction')
        
        # Generate ROC curve
        if self.config['evaluation']['visualization']['roc_curve']:
            self._plot_roc_curve(y_test, y_pred_proba, 'health_prediction')
        
        # Generate model explanations
        if self.config['evaluation']['explanation']['shap']:
            if not isinstance(model, list):
                self._generate_shap_explanations(model, X_test, 'health_prediction')
        
        if self.config['evaluation']['explanation']['lime']:
            if not isinstance(model, list):
                self._generate_lime_explanations(model, X_test, y_test, 'health_prediction')
        
        logger.info(f"Health prediction model evaluation complete. Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics.get('auc', 'N/A')}")
        
        return metrics
    
    def _get_ensemble_predictions(self, models, X):
        """Get predictions from ensemble models with weighted averaging."""
        ensemble_method = self.config['ensemble']['method']
        weights = self.config['ensemble']['weights']
        
        # If weights are not specified, use equal weights
        if not weights or len(weights) != len(models):
            weights = [1/len(models)] * len(models)
        
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        if ensemble_method == 'voting':
            # For voting, get class predictions and use majority vote
            predictions = []
            for i, model in enumerate(models):
                y_pred = model.predict(X)
                if y_pred.shape[1] > 1:  # Multi-class
                    predictions.append(np.argmax(y_pred, axis=1))
                else:  # Binary
                    predictions.append((y_pred > 0.5).astype(int).ravel())
            
            # Stack predictions and take mode along axis 0
            stacked = np.stack(predictions)
            y_pred = np.apply_along_axis(lambda x: np.bincount(x, weights=weights).argmax(), axis=0, arr=stacked)
            
            # Convert back to one-hot if multi-class
            if models[0].output_shape[-1] > 1:
                y_pred = to_categorical(y_pred, num_classes=models[0].output_shape[-1])
            else:
                y_pred = y_pred.reshape(-1, 1)
            
            return y_pred
        
        else:  # Default to averaging probabilities
            # For averaging, get probability predictions and take weighted average
            predictions = []
            for i, model in enumerate(models):
                predictions.append(model.predict(X) * weights[i])
            
            # Sum weighted predictions
            y_pred = np.sum(predictions, axis=0)
            
            return y_pred
