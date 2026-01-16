"""
Hybrid Model combining CNN, RNN, and Autoencoders for maximum accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from .cnn_model import CNNDeepfakeDetector
from .rnn_model import RNNDeepfakeDetector


class HybridDeepfakeDetector(models.Model):
    """
    Hybrid model combining:
    - CNN for spatial feature extraction
    - RNN for temporal analysis
    - Attention mechanisms for focus areas
    """
    
    def __init__(self, image_shape=(256, 256, 3), sequence_length=8):
        super(HybridDeepfakeDetector, self).__init__()
        self.image_shape = image_shape
        self.sequence_length = sequence_length
        
        # CNN branch for spatial features
        self.cnn = CNNDeepfakeDetector(input_shape=image_shape)
        self.cnn_output_dim = 512  # Output from CNN features
        
        # RNN branch for temporal features
        self.rnn = RNNDeepfakeDetector(frame_features_dim=512, sequence_length=sequence_length)
        
        # Feature fusion layers
        self.fusion_dense1 = layers.Dense(256, activation='relu')
        self.fusion_bn1 = layers.BatchNormalization()
        self.fusion_dropout1 = layers.Dropout(0.3)
        
        self.fusion_dense2 = layers.Dense(128, activation='relu')
        self.fusion_bn2 = layers.BatchNormalization()
        self.fusion_dropout2 = layers.Dropout(0.2)
        
        # Multi-head attention for feature importance
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        
        # Final classification
        self.final_dense1 = layers.Dense(64, activation='relu')
        self.final_dropout = layers.Dropout(0.2)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x_frames, x_sequences=None, training=False):
        """
        Args:
            x_frames: Individual frames for CNN (batch_size, height, width, 3)
            x_sequences: Frame sequence features for RNN (batch_size, seq_len, feature_dim)
        """
        # CNN spatial analysis
        cnn_features = []
        if isinstance(x_frames, list):
            for frame in x_frames:
                feat = self.cnn(tf.expand_dims(frame, 0), training=training)
                cnn_features.append(feat)
            cnn_out = tf.concat(cnn_features, axis=0)
        else:
            # Extract multiple frames if single input
            cnn_out = self.cnn(x_frames, training=training)
        
        # RNN temporal analysis (if sequences provided)
        if x_sequences is not None:
            rnn_out = self.rnn(x_sequences, training=training)
        else:
            # Use CNN features as temporal sequence
            rnn_out = self.rnn(tf.expand_dims(cnn_out, 1), training=training)
        
        # Combine CNN and RNN features
        combined = tf.concat([cnn_out, tf.expand_dims(rnn_out, 1)], axis=-1)
        
        # Fusion with attention
        if len(combined.shape) > 2:
            attended = self.attention(combined, combined, training=training)
        else:
            attended = combined
        
        # Dense fusion layers
        x = self.fusion_dense1(attended if len(attended.shape) > 1 else tf.expand_dims(attended, 1))
        x = self.fusion_bn1(x, training=training)
        x = self.fusion_dropout1(x, training=training)
        
        x = self.fusion_dense2(x)
        x = self.fusion_bn2(x, training=training)
        x = self.fusion_dropout2(x, training=training)
        
        # Flatten if needed
        if len(x.shape) > 2:
            x = tf.reduce_mean(x, axis=1)
        
        # Final classification
        x = self.final_dense1(x)
        x = self.final_dropout(x, training=training)
        x = self.output_layer(x)
        
        return x


class EnsembleDeepfakeDetector(models.Model):
    """Ensemble of multiple models for robust detection"""
    
    def __init__(self, models_list):
        super(EnsembleDeepfakeDetector, self).__init__()
        self.models_list = models_list
        self.num_models = len(models_list)
        
        # Learnable weights for each model
        self.model_weights = self.add_weight(
            name='model_weights',
            shape=(len(models_list),),
            initializer=keras.initializers.Constant(1.0 / len(models_list)),
            trainable=True
        )
    
    def call(self, x, training=False):
        predictions = []
        for model in self.models_list:
            pred = model(x, training=training)
            predictions.append(pred)
        
        # Weighted ensemble
        predictions = tf.stack(predictions, axis=1)
        weights = tf.nn.softmax(self.model_weights)
        weighted_pred = tf.reduce_sum(predictions * weights, axis=1)
        
        return weighted_pred


def create_hybrid_model(image_shape=(256, 256, 3), sequence_length=8):
    """Factory function to create hybrid model"""
    model = HybridDeepfakeDetector(image_shape, sequence_length)
    model.build(input_shape=[(None, *image_shape), (None, sequence_length, 512)])
    return model
