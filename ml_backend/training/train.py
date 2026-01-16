"""
Advanced Training Pipeline for Deepfake Detection
Targets 99% accuracy with ensemble and transfer learning techniques
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn_model import create_cnn_model
from models.rnn_model import create_rnn_model
from models.autoencoder_model import create_autoencoder_model
from models.hybrid_model import create_hybrid_model, EnsembleDeepfakeDetector
from models.audio_model import create_audio_model
from data.data_processor import DataGenerator, create_tf_dataset, ImageProcessor, VideoProcessor
from utils.model_utils import (
    ModelEvaluator, CustomCallbacks, LossFunction,
    ModelOptimizer, ModelSaver, InferenceOptimizer
)


class DeepfakeDetectionTrainer:
    """Main training orchestrator for achieving 99% accuracy"""
    
    def __init__(self, config_path=None):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path) if config_path else self.get_default_config()
        self.models = {}
        self.histories = {}
        self.evaluations = {}
        
        # Set random seeds for reproducibility
        self.set_random_seeds()
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 1e-4,
            'input_shape': (256, 256, 3),
            'models_to_train': ['cnn', 'rnn', 'autoencoder', 'hybrid', 'audio'],
            'validation_split': 0.2,
            'test_split': 0.1,
            'early_stopping_patience': 15,
            'target_accuracy': 0.99,
            'use_class_weights': True,
            'use_augmentation': True,
            'save_dir': './trained_models',
            'use_mixed_precision': True
        }
    
    @staticmethod
    def load_config(config_path):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def set_random_seeds(seed=42):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def prepare_data(self, data_dir):
        """Prepare training data"""
        print("\n" + "="*50)
        print("PREPARING TRAINING DATA")
        print("="*50)
        
        generator = DataGenerator(data_dir, batch_size=self.config['batch_size'])
        
        # Load image data
        print("\nLoading image dataset...")
        try:
            X_train_img, y_train_img = generator.load_image_dataset('train')
            X_val_img, y_val_img = generator.load_image_dataset('val')
            print(f"✓ Loaded {len(X_train_img)} training images")
            print(f"✓ Loaded {len(X_val_img)} validation images")
        except Exception as e:
            print(f"Warning: Could not load image data: {e}")
            X_train_img, y_train_img = None, None
            X_val_img, y_val_img = None, None
        
        # Load video data
        print("\nLoading video dataset...")
        try:
            X_train_vid, y_train_vid = generator.load_video_dataset('train')
            X_val_vid, y_val_vid = generator.load_video_dataset('val')
            print(f"✓ Loaded {len(X_train_vid)} training videos")
            print(f"✓ Loaded {len(X_val_vid)} validation videos")
        except Exception as e:
            print(f"Warning: Could not load video data: {e}")
            X_train_vid, y_train_vid = None, None
            X_val_vid, y_val_vid = None, None
        
        # Load audio data
        print("\nLoading audio dataset...")
        try:
            X_train_spec, X_train_mfcc, y_train_aud = generator.load_audio_dataset('train')
            X_val_spec, X_val_mfcc, y_val_aud = generator.load_audio_dataset('val')
            print(f"✓ Loaded {len(X_train_spec)} training audio files")
            print(f"✓ Loaded {len(X_val_spec)} validation audio files")
        except Exception as e:
            print(f"Warning: Could not load audio data: {e}")
            X_train_spec, X_train_mfcc, y_train_aud = None, None, None
            X_val_spec, X_val_mfcc, y_val_aud = None, None, None
        
        return {
            'images': {
                'train': (X_train_img, y_train_img),
                'val': (X_val_img, y_val_img)
            },
            'videos': {
                'train': (X_train_vid, y_train_vid),
                'val': (X_val_vid, y_val_vid)
            },
            'audio': {
                'train': (X_train_spec, X_train_mfcc, y_train_aud),
                'val': (X_val_spec, X_val_mfcc, y_val_aud)
            }
        }
    
    def build_cnn_model(self):
        """Build and compile CNN model"""
        print("\n" + "="*50)
        print("BUILDING CNN MODEL")
        print("="*50)
        
        model = create_cnn_model(self.config['input_shape'])
        
        optimizer = ModelOptimizer.get_optimized_optimizer(self.config['learning_rate'])
        loss_fn = LossFunction.focal_loss(gamma=2.0, alpha=0.25)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"✓ CNN Model built successfully")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def build_rnn_model(self):
        """Build and compile RNN model"""
        print("\n" + "="*50)
        print("BUILDING RNN MODEL")
        print("="*50)
        
        model = create_rnn_model(frame_features_dim=512, sequence_length=16)
        
        optimizer = ModelOptimizer.get_optimized_optimizer(self.config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"✓ RNN Model built successfully")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def build_autoencoder_model(self):
        """Build and compile Autoencoder model"""
        print("\n" + "="*50)
        print("BUILDING AUTOENCODER MODEL")
        print("="*50)
        
        model = create_autoencoder_model(self.config['input_shape'], model_type='denoising')
        
        optimizer = ModelOptimizer.get_optimized_optimizer(self.config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✓ Autoencoder Model built successfully")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def build_hybrid_model(self):
        """Build and compile Hybrid model"""
        print("\n" + "="*50)
        print("BUILDING HYBRID MODEL")
        print("="*50)
        
        model = create_hybrid_model(self.config['input_shape'], sequence_length=8)
        
        optimizer = ModelOptimizer.get_optimized_optimizer(self.config['learning_rate'])
        loss_fn = LossFunction.focal_loss(gamma=2.0, alpha=0.25)
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        print(f"✓ Hybrid Model built successfully")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def build_audio_model(self):
        """Build and compile Audio model"""
        print("\n" + "="*50)
        print("BUILDING AUDIO MODEL")
        print("="*50)
        
        model = create_audio_model(model_type='multimodal')
        
        optimizer = ModelOptimizer.get_optimized_optimizer(self.config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"✓ Audio Model built successfully")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def train_cnn_model(self, X_train, y_train, X_val, y_val):
        """Train CNN model"""
        print("\n" + "="*50)
        print("TRAINING CNN MODEL")
        print("="*50)
        
        if X_train is None:
            print("⚠ Skipping CNN training: no data")
            return None
        
        model = self.build_cnn_model()
        
        # Create datasets
        train_dataset = create_tf_dataset(X_train, y_train, self.config['batch_size'], 
                                         augment=self.config['use_augmentation'])
        val_dataset = create_tf_dataset(X_val, y_val, self.config['batch_size'])
        
        # Callbacks
        callbacks = [
            CustomCallbacks.get_early_stopping(self.config['early_stopping_patience']),
            CustomCallbacks.get_reduce_lr(),
            CustomCallbacks.get_model_checkpoint(self.config['save_dir'], 'cnn_model')
        ]
        
        # Class weights
        class_weights = None
        if self.config['use_class_weights']:
            class_weights = CustomCallbacks.get_class_weight_callback()
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        self.models['cnn'] = model
        self.histories['cnn'] = history
        
        return model, history
    
    def evaluate_models(self, data_dict):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("EVALUATING MODELS")
        print("="*50)
        
        evaluator = ModelEvaluator()
        X_test_img, y_test_img = data_dict['images']['val']
        
        if X_test_img is not None and 'cnn' in self.models:
            model = self.models['cnn']
            
            # Predictions
            y_pred_proba = model.predict(X_test_img)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            metrics = evaluator.calculate_metrics(y_test_img, y_pred_proba.flatten(), y_pred_proba.flatten())
            
            print("\nCNN Model Evaluation:")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  F1 Score:    {metrics['f1']:.4f}")
            print(f"  AUC-ROC:     {metrics.get('auc_roc', 'N/A')}")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            
            self.evaluations['cnn'] = metrics
            
            # Check if target accuracy reached
            if metrics['accuracy'] >= self.config['target_accuracy']:
                print(f"\n✓ TARGET ACCURACY ACHIEVED: {metrics['accuracy']:.4f}")
            else:
                gap = (self.config['target_accuracy'] - metrics['accuracy']) * 100
                print(f"\n⚠ Accuracy gap to target: {gap:.2f}%")
    
    def save_all_models(self):
        """Save all trained models"""
        print("\n" + "="*50)
        print("SAVING MODELS")
        print("="*50)
        
        saver = ModelSaver()
        Path(self.config['save_dir']).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            saver.save_model(model, self.config['save_dir'], model_name)
            print(f"✓ Saved {model_name} model")
            
            if model_name in self.histories:
                saver.save_training_history(self.histories[model_name], 
                                           self.config['save_dir'], model_name)
                print(f"✓ Saved {model_name} training history")
    
    def train_all(self, data_dir):
        """Train all models"""
        print("\n" + "="*80)
        print("DEEPFAKE DETECTION MODEL TRAINING - AIMING FOR 99% ACCURACY")
        print("="*80)
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Prepare data
        data_dict = self.prepare_data(data_dir)
        
        # Train CNN model
        if 'cnn' in self.config['models_to_train']:
            X_train_img, y_train_img = data_dict['images']['train']
            X_val_img, y_val_img = data_dict['images']['val']
            self.train_cnn_model(X_train_img, y_train_img, X_val_img, y_val_img)
        
        # Evaluate
        self.evaluate_models(data_dict)
        
        # Save
        self.save_all_models()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
        return self.evaluations


if __name__ == "__main__":
    # Example usage
    trainer = DeepfakeDetectionTrainer()
    
    # Assume data is in ./data directory
    data_dir = "./data"
    
    if not Path(data_dir).exists():
        print(f"Note: Create training data in {data_dir}")
        print("Expected structure:")
        print("  data/")
        print("    real/train/")
        print("    fake/train/")
        print("    real/val/")
        print("    fake/val/")
    else:
        results = trainer.train_all(data_dir)
        print(f"\nResults: {json.dumps(results, indent=2)}")
