"""
Main module initialization
"""

__version__ = '1.0.0'
__author__ = 'DeepScan Team'

# Models
from models.cnn_model import create_cnn_model, CNNDeepfakeDetector
from models.rnn_model import create_rnn_model, RNNDeepfakeDetector, GRUDeepfakeDetector
from models.autoencoder_model import create_autoencoder_model
from models.hybrid_model import create_hybrid_model, HybridDeepfakeDetector, EnsembleDeepfakeDetector
from models.audio_model import create_audio_model

# Data processing
from data.data_processor import (
    ImageProcessor, VideoProcessor, AudioProcessor,
    DataGenerator, create_tf_dataset
)

# Utilities
from utils.model_utils import (
    ModelEvaluator, CustomCallbacks, LossFunction,
    ModelOptimizer, ModelSaver, InferenceOptimizer
)

# Training
from training.train import DeepfakeDetectionTrainer

# Inference
from inference.engine import DeepfakeInferenceEngine, get_inference_engine

__all__ = [
    'create_cnn_model',
    'create_rnn_model',
    'create_autoencoder_model',
    'create_hybrid_model',
    'create_audio_model',
    'ImageProcessor',
    'VideoProcessor',
    'AudioProcessor',
    'DataGenerator',
    'DeepfakeDetectionTrainer',
    'DeepfakeInferenceEngine',
    'get_inference_engine',
    'ModelEvaluator',
    'CustomCallbacks',
    'LossFunction'
]
