"""
Model utilities for training, evaluation, and inference
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import json
from pathlib import Path
from datetime import datetime


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred_binary)),
            'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred_binary, zero_division=0)),
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred_proba))
        
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        metrics['specificity'] = float(metrics['tn'] / (metrics['tn'] + metrics['fp'] + 1e-8))
        metrics['sensitivity'] = float(metrics['tp'] / (metrics['tp'] + metrics['fn'] + 1e-8))
        
        return metrics
    
    @staticmethod
    def get_roc_curve(y_true, y_pred):
        """Get ROC curve data"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }


class CustomCallbacks:
    """Custom callbacks for training"""
    
    @staticmethod
    def get_early_stopping(patience=10):
        """Early stopping callback"""
        return keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    
    @staticmethod
    def get_reduce_lr(factor=0.5, patience=5):
        """Learning rate reduction callback"""
        return keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience,
            verbose=1,
            min_lr=1e-6
        )
    
    @staticmethod
    def get_model_checkpoint(save_dir, model_name):
        """Model checkpoint callback"""
        save_path = Path(save_dir) / f"{model_name}_best.h5"
        return keras.callbacks.ModelCheckpoint(
            str(save_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    
    @staticmethod
    def get_class_weight_callback():
        """Calculate class weights"""
        # For imbalanced data
        return {0: 1.0, 1: 1.5}


class LossFunction:
    """Custom loss functions for deepfake detection"""
    
    @staticmethod
    def focal_loss(gamma=2., alpha=0.25):
        """Focal loss to handle class imbalance"""
        def focal_loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            epsilon = keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            modulating_factor = tf.pow(1. - p_t, gamma)
            loss = -alpha * modulating_factor * tf.math.log(p_t)
            
            return tf.reduce_mean(loss)
        
        return focal_loss_fn
    
    @staticmethod
    def weighted_binary_crossentropy(weight_positive=1.5):
        """Weighted binary crossentropy"""
        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = keras.losses.binary_crossentropy(y_true, y_pred)
            weights = y_true * weight_positive + (1 - y_true) * 1.0
            return tf.reduce_mean(bce * weights)
        
        return loss_fn


class ModelOptimizer:
    """Optimization strategies for achieving high accuracy"""
    
    @staticmethod
    def get_optimized_optimizer(learning_rate=1e-4, beta_1=0.9, beta_2=0.999):
        """Get optimized Adam optimizer"""
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-7
        )
    
    @staticmethod
    def get_scheduled_learning_rate(initial_lr=1e-4, decay_steps=1000, decay_rate=0.96):
        """Learning rate schedule"""
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )


class ModelSaver:
    """Save and load models"""
    
    @staticmethod
    def save_model(model, save_dir, model_name):
        """Save model"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / f"{model_name}.h5"
        model.save(str(model_path))
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'model_type': str(type(model).__name__)
        }
        
        metadata_path = save_path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(model_path)
    
    @staticmethod
    def load_model(model_path):
        """Load model"""
        return keras.models.load_model(str(model_path))
    
    @staticmethod
    def save_training_history(history, save_dir, model_name):
        """Save training history"""
        save_path = Path(save_dir) / f"{model_name}_history.json"
        
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)


class InferenceOptimizer:
    """Optimize inference speed"""
    
    @staticmethod
    def convert_to_tflite(model, save_path):
        """Convert model to TFLite for faster inference"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
    
    @staticmethod
    def quantize_model(model):
        """Quantize model for faster inference"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        return converter.convert()
