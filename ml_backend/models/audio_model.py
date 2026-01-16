"""
Audio Analysis Model for Deepfake Detection
Detects synthesized/manipulated audio
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


class AudioSpectrumAnalyzer(models.Model):
    """Analyzes audio spectrogram for deepfake detection"""
    
    def __init__(self, input_shape=(256, 256, 1)):
        super(AudioSpectrumAnalyzer, self).__init__()
        self.input_shape_val = input_shape
        
        # Spectrogram analysis
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        # LSTM for temporal patterns in spectrogram
        self.lstm = layers.LSTM(128, return_sequences=False)
        
        # Classification
        self.dropout = layers.Dropout(0.3)
        self.dense = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        
        # Reshape for LSTM
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1] * shape[2], shape[3]))
        
        # LSTM processing
        x = self.lstm(x, training=training)
        
        # Classification
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.output_layer(x)
        
        return x


class MFCCAnalyzer(models.Model):
    """Analyzes MFCC features for voice deepfake detection"""
    
    def __init__(self, input_shape=(128, 44), n_mfcc=40):
        super(MFCCAnalyzer, self).__init__()
        self.input_shape_val = input_shape
        self.n_mfcc = n_mfcc
        
        # MFCC feature extraction layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        # Temporal modeling
        self.gru = layers.GRU(128, return_sequences=True)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        
        # Dense layers
        self.dropout1 = layers.Dropout(0.3)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        # Conv processing
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        # Reshape for temporal processing
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1], shape[2] * shape[3]))
        
        # GRU + Attention
        x = self.gru(x, training=training)
        x = self.attention(x, x, training=training)
        
        # Global pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Dense classification
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        x = self.output_layer(x)
        
        return x


class VoiceConsistencyAnalyzer(models.Model):
    """Analyzes voice consistency and naturalness"""
    
    def __init__(self):
        super(VoiceConsistencyAnalyzer, self).__init__()
        
        # Pitch analysis
        self.pitch_encoder = layers.Dense(64, activation='relu')
        
        # Energy analysis
        self.energy_encoder = layers.Dense(64, activation='relu')
        
        # Formant analysis
        self.formant_encoder = layers.Dense(64, activation='relu')
        
        # Combine
        self.fusion = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.output = layers.Dense(1, activation='sigmoid')
    
    def call(self, pitch_features, energy_features, formant_features, training=False):
        # Encode each component
        pitch = self.pitch_encoder(pitch_features)
        energy = self.energy_encoder(energy_features)
        formants = self.formant_encoder(formant_features)
        
        # Combine
        combined = tf.concat([pitch, energy, formants], axis=-1)
        x = self.fusion(combined)
        x = self.dropout(x, training=training)
        x = self.output(x)
        
        return x


class MultimodalAudioDetector(models.Model):
    """Combines multiple audio analysis techniques"""
    
    def __init__(self):
        super(MultimodalAudioDetector, self).__init__()
        
        self.spectrogram_analyzer = AudioSpectrumAnalyzer()
        self.mfcc_analyzer = MFCCAnalyzer()
        
        # Fusion layer
        self.fusion_dense = layers.Dense(128, activation='relu')
        self.fusion_dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, spectrogram, mfcc, training=False):
        # Analyze both representations
        spec_pred = self.spectrogram_analyzer(spectrogram, training=training)
        mfcc_pred = self.mfcc_analyzer(mfcc, training=training)
        
        # Combine predictions
        combined = tf.concat([spec_pred, mfcc_pred], axis=-1)
        x = self.fusion_dense(combined)
        x = self.fusion_dropout(x, training=training)
        x = self.output_layer(x)
        
        return x


def create_audio_model(model_type='spectrum'):
    """Factory function to create audio model"""
    if model_type == 'mfcc':
        model = MFCCAnalyzer()
    elif model_type == 'multimodal':
        model = MultimodalAudioDetector()
    else:  # spectrum
        model = AudioSpectrumAnalyzer()
    
    return model
