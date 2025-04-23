"""
CYBI Smartshoe AI System
========================

A comprehensive AI system for predicting human weight and health conditions
from CYBI smartshoe sensor data.

Components:
- data_module: Advanced data processing and feature extraction
- model_architecture: Deep learning model architectures
- training_module: Training and evaluation
- main: System integration and main API
- deploy_api: REST API for deployment
"""

__version__ = '1.0.0'

# Import key components for easy access
from cybi.data_module import CYBIDataProcessor
from cybi.model_architecture import CYBIModelBuilder
from cybi.main import CYBISystem

# Define package exports
__all__ = [
    'CYBIDataProcessor',
    'CYBIModelBuilder',
    'CYBISystem',
] 