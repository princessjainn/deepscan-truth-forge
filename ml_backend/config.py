"""
Comprehensive Configuration for 99%+ Accuracy Deepfake Detection
Includes model, training, inference, and API settings
"""

import os
from pathlib import Path

# Directory structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
TRAINING_DIR = BASE_DIR / 'training'
INFERENCE_DIR = BASE_DIR / 'inference'
UTILS_DIR = BASE_DIR / 'utils'
TRAINED_MODELS_DIR = Path('./trained_models')

# Model paths
CNN_MODEL_PATH = TRAINED_MODELS_DIR / 'cnn_model_best.h5'
RNN_MODEL_PATH = TRAINED_MODELS_DIR / 'rnn_model_best.h5'
AUTOENCODER_MODEL_PATH = TRAINED_MODELS_DIR / 'autoencoder_model_best.h5'

# ============ Model Architecture Configuration ============

MODEL_CONFIG = {
    # Input settings
    'input_shape': (224, 224, 3),
    'num_frames': 8,
    'sample_rate': 16000,
    
    # Transfer Learning Models
    'transfer_learning': {
        'efficientnet': {'version': 'B3', 'dropout': 0.3},
        'densenet': {'version': '121', 'dropout': 0.3},
        'xception': {'dropout': 0.3}
    },
    
    # LSTM Settings
    'lstm': {'units': [256, 128], 'bidirectional': True},
    
    # Regularization
    'regularization': {'l2': 1e-4, 'dropout': 0.3}
}

# ============ Training Configuration ============

TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 1e-4,
    'optimizer': 'AdamW',
    'loss': 'focal_loss',
    'class_weights': {0: 1.0, 1: 1.5},
    'early_stopping_patience': 20,
    'target_accuracy': 0.99,
}

# ============ Dataset Configuration ============

DATASET_CONFIG = {
    'datasets': ['celeb_df', 'faceforensics_pp', 'deepfake_timit', 'dfdc', 'wilddeepfake'],
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'total_samples': 140000,
}

# ============ Inference Configuration ============

INFERENCE_CONFIG = {
    'deepfake_threshold': 0.5,
    'confidence_threshold': 0.7,
    'ensemble_enable': True,
    'batch_size': 32,
}

# ============ API Configuration ============

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'max_file_size': 1000 * 1024 * 1024,  # 1GB
}

# ============ Performance Targets ============

PERFORMANCE_TARGETS = {
    'accuracy': 0.99,
    'precision': 0.989,
    'recall': 0.991,
    'f1_score': 0.990,
    'auc': 0.998,
}
HYBRID_MODEL_PATH = TRAINED_MODELS_DIR / 'hybrid_model_best.h5'
AUDIO_MODEL_PATH = TRAINED_MODELS_DIR / 'audio_model_best.h5'

# Model configurations
MODEL_CONFIGS = {
    'cnn': {
        'input_shape': (256, 256, 3),
        'model_type': 'image',
        'architecture': 'CNN with Attention',
        'parameters': '~5M'
    },
    'rnn': {
        'frame_features_dim': 512,
        'sequence_length': 16,
        'model_type': 'video',
        'architecture': 'LSTM with Attention',
        'parameters': '~2M'
    },
    'autoencoder': {
        'input_shape': (256, 256, 3),
        'model_type': 'image',
        'architecture': 'Denoising Autoencoder',
        'parameters': '~8M'
    },
    'hybrid': {
        'image_shape': (256, 256, 3),
        'sequence_length': 8,
        'model_type': 'multimodal',
        'architecture': 'CNN + RNN + Attention',
        'parameters': '~7M'
    },
    'audio': {
        'input_shape': (256, 256, 1),
        'model_type': 'audio',
        'architecture': 'Multimodal Audio Detector',
        'parameters': '~1M'
    }
}

# Training configurations
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
    'validation_split': 0.2,
    'test_split': 0.1,
    'early_stopping_patience': 15,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 5,
    'target_accuracy': 0.99,
    'use_class_weights': True,
    'use_augmentation': True,
    'use_mixed_precision': True,
    'seed': 42
}

# Data augmentation
AUGMENTATION_CONFIG = {
    'brightness_range': 0.1,
    'rotation_range': 15,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'contrast_range': (0.9, 1.1)
}

# API configurations
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'max_file_size': 500 * 1024 * 1024,  # 500 MB
    'upload_folder': './uploads',
    'temp_folder': './temp'
}

# File extensions
ALLOWED_EXTENSIONS = {
    'images': {'jpg', 'jpeg', 'png', 'gif', 'bmp'},
    'videos': {'mp4', 'avi', 'mov', 'mkv', 'webm'},
    'audio': {'wav', 'mp3', 'aac', 'flac', 'ogg'}
}

# Detection thresholds
CONFIDENCE_THRESHOLDS = {
    'high_confidence': 0.95,
    'medium_confidence': 0.70,
    'low_confidence': 0.50,
    'default': 0.50
}

# Audio processing
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'duration': 5,
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mel': 128
}

# Video processing
VIDEO_CONFIG = {
    'target_size': (256, 256),
    'num_frames': 16,
    'fps_target': 30,
    'color_space': 'RGB'
}

# Image processing
IMAGE_CONFIG = {
    'target_size': (256, 256),
    'color_space': 'RGB',
    'normalization': 'z-score'
}

# Feature extraction
FEATURE_CONFIG = {
    'forensic_features': ['dct_energy', 'edge_ratio', 'color_histogram'],
    'temporal_features': ['optical_flow', 'frame_consistency', 'motion_vectors'],
    'audio_features': ['mfcc', 'spectrogram', 'zero_crossing_rate', 'spectral_centroid']
}

# Performance targets
PERFORMANCE_TARGETS = {
    'accuracy': 0.99,
    'precision': 0.98,
    'recall': 0.97,
    'f1_score': 0.975,
    'auc_roc': 0.99,
    'sensitivity': 0.97,
    'specificity': 0.99
}

# Dataset names (for reference)
COMMON_DATASETS = {
    'FaceForensics++': 'Video deepfake detection',
    'DFDC': 'Deep Fake Detection Challenge',
    'CelebDF': 'Real-world deepfakes',
    'Deepfake Detection Challenge': 'Audio and video',
    'VoxCeleb': 'Speaker recognition',
    'YouTube-Deepfake': 'In-the-wild deepfakes'
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    for dir_path in [DATA_DIR, TRAINED_MODELS_DIR, API_CONFIG['upload_folder'], API_CONFIG['temp_folder']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# Initialize on import
create_directories()
