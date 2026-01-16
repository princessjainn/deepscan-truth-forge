"""
LSTM/RNN Model for Temporal Deepfake Detection
Analyzes temporal inconsistencies in video sequences
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class TemporalAttentionLSTM(layers.Layer):
    """LSTM with attention for temporal sequence analysis"""
    
    def __init__(self, units=128):
        super(TemporalAttentionLSTM, self).__init__()
        self.units = units
        self.lstm = layers.LSTM(units, return_sequences=True)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=units // 4)
        self.dense = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        lstm_out = self.lstm(x, training=training)
        attention_out = self.attention(lstm_out, lstm_out, training=training)
        return attention_out


class RNNDeepfakeDetector(models.Model):
    """Advanced RNN model for temporal deepfake detection"""
    
    def __init__(self, frame_features_dim=512, sequence_length=16):
        super(RNNDeepfakeDetector, self).__init__()
        self.frame_features_dim = frame_features_dim
        self.sequence_length = sequence_length
        
        # Feature normalization
        self.norm = layers.LayerNormalization()
        
        # Temporal attention blocks
        self.temporal_lstm1 = TemporalAttentionLSTM(units=256)
        self.dropout1 = layers.Dropout(0.3)
        
        self.temporal_lstm2 = TemporalAttentionLSTM(units=128)
        self.dropout2 = layers.Dropout(0.2)
        
        # Bidirectional processing
        self.bidirectional = layers.Bidirectional(
            layers.LSTM(64, return_sequences=False)
        )
        
        # Dense classifier
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout3 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        # Input shape: (batch_size, sequence_length, frame_features_dim)
        x = self.norm(x)
        
        # First temporal attention block
        x = self.temporal_lstm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        # Second temporal attention block
        x = self.temporal_lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Bidirectional processing
        x = self.bidirectional(x, training=training)
        
        # Classification
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        x = self.dense2(x)
        x = self.output_layer(x)
        
        return x
    
    def get_config(self):
        return {
            "frame_features_dim": self.frame_features_dim,
            "sequence_length": self.sequence_length
        }


class GRUDeepfakeDetector(models.Model):
    """GRU-based model (faster alternative to LSTM)"""
    
    def __init__(self, frame_features_dim=512, sequence_length=16):
        super(GRUDeepfakeDetector, self).__init__()
        self.frame_features_dim = frame_features_dim
        self.sequence_length = sequence_length
        
        # Feature normalization
        self.norm = layers.LayerNormalization()
        
        # GRU layers
        self.gru1 = layers.GRU(256, return_sequences=True)
        self.dropout1 = layers.Dropout(0.3)
        
        self.gru2 = layers.GRU(128, return_sequences=True)
        self.dropout2 = layers.Dropout(0.2)
        
        self.gru3 = layers.GRU(64, return_sequences=False)
        
        # Dense layers
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout3 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        x = self.norm(x)
        
        x = self.gru1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.gru2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.gru3(x, training=training)
        
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        x = self.dense2(x)
        x = self.output_layer(x)
        
        return x


def create_rnn_model(frame_features_dim=512, sequence_length=16, use_gru=False):
    """Factory function to create RNN model"""
    if use_gru:
        model = GRUDeepfakeDetector(frame_features_dim, sequence_length)
    else:
        model = RNNDeepfakeDetector(frame_features_dim, sequence_length)
    model.build(input_shape=(None, sequence_length, frame_features_dim))
    return model
