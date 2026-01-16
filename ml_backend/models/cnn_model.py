"""
Advanced CNN Model for Deepfake Detection
Uses multiple convolutional layers with attention mechanisms
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


class AttentionBlock(layers.Layer):
    """Attention mechanism for focusing on critical regions"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.query_conv = layers.Conv2D(channels // 8, kernel_size=1)
        self.key_conv = layers.Conv2D(channels // 8, kernel_size=1)
        self.value_conv = layers.Conv2D(channels, kernel_size=1)
        self.gamma = self.add_weight(
            name='gamma',
            shape=[1],
            initializer=keras.initializers.Constant(0.0),
            trainable=True
        )
    
    def call(self, x):
        batch_size, height, width, channels = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        
        # Project input
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Reshape for attention computation
        query = tf.reshape(query, [batch_size, -1, self.channels // 8])
        key = tf.reshape(key, [batch_size, -1, self.channels // 8])
        value = tf.reshape(value, [batch_size, -1, self.channels])
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.channels // 8, tf.float32))
        attention = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        out = tf.matmul(attention, value)
        out = tf.reshape(out, [batch_size, height, width, channels])
        
        # Residual connection with learned weight
        return x + self.gamma * out


class CNNDeepfakeDetector(models.Model):
    """Advanced CNN model for deepfake detection"""
    
    def __init__(self, input_shape=(256, 256, 3)):
        super(CNNDeepfakeDetector, self).__init__()
        self.input_shape_val = input_shape
        
        # Feature extraction blocks
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        self.conv2 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.attention2 = AttentionBlock(128)
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2)
        
        self.conv3 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.attention3 = AttentionBlock(256)
        self.pool3 = layers.MaxPooling2D(pool_size=2, strides=2)
        
        self.conv4 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.attention4 = AttentionBlock(512)
        self.pool4 = layers.MaxPooling2D(pool_size=2, strides=2)
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Dense layers
        self.dropout1 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(512, activation='relu')
        self.bn_dense1 = layers.BatchNormalization()
        
        self.dropout2 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.bn_dense2 = layers.BatchNormalization()
        
        self.dropout3 = layers.Dropout(0.2)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.attention2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.attention3(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.attention4(x)
        x = self.pool4(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Dense layers
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.bn_dense1(x, training=training)
        
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        x = self.bn_dense2(x, training=training)
        
        x = self.dropout3(x, training=training)
        x = self.output_layer(x)
        
        return x
    
    def get_config(self):
        return {"input_shape": self.input_shape_val}


def create_cnn_model(input_shape=(256, 256, 3)):
    """Factory function to create CNN model"""
    model = CNNDeepfakeDetector(input_shape=input_shape)
    model.build(input_shape=(None, *input_shape))
    return model
