"""
Autoencoder Model for Deepfake Detection
Learns to reconstruct authentic media and detects anomalies
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class DenosingAutoencoder(models.Model):
    """Denoising Autoencoder for anomaly detection"""
    
    def __init__(self, input_shape=(256, 256, 3)):
        super(DenosingAutoencoder, self).__init__()
        self.input_shape_val = input_shape
        
        # Encoder
        self.encoder = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
        ])
        
        # Decoder
        self.decoder = models.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(8192, activation='relu'),
            layers.Reshape((32, 32, 8)),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
        ])
    
    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)
    
    def get_config(self):
        return {"input_shape": self.input_shape_val}


class VariationalAutoencoder(models.Model):
    """Variational Autoencoder for generative deepfake detection"""
    
    def __init__(self, input_shape=(256, 256, 3), latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        self.input_shape_val = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
        ])
        
        # Latent space
        self.mean_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = models.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(32 * 32 * 128, activation='relu'),
            layers.Reshape((32, 32, 128)),
            layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same'),
        ])
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mean + eps * std
    
    def call(self, x, training=False):
        # Encode
        encoded = self.encoder(x, training=training)
        mean = self.mean_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Sample from latent space
        z = self.reparameterize(mean, logvar)
        
        # Decode
        decoded = self.decoder(z, training=training)
        
        return decoded, mean, logvar
    
    def encode(self, x):
        """Get latent representation"""
        encoded = self.encoder(x)
        mean = self.mean_layer(encoded)
        return mean
    
    def get_config(self):
        return {
            "input_shape": self.input_shape_val,
            "latent_dim": self.latent_dim
        }


class AnomalyDetectionAutoencoder(models.Model):
    """Specialized autoencoder for anomaly/manipulation detection"""
    
    def __init__(self, input_shape=(256, 256, 3)):
        super(AnomalyDetectionAutoencoder, self).__init__()
        self.input_shape_val = input_shape
        
        # Lightweight encoder for fast anomaly detection
        self.encoder = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (5, 5), strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (5, 5), strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
        ])
        
        # Decoder with upsampling
        self.decoder = models.Sequential([
            layers.Dense(64 * 64 * 32, activation='relu'),
            layers.Reshape((64, 64, 32)),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
        ])
    
    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        encoded = tf.expand_dims(encoded, 0)  # Add dimension for decoder
        decoded = self.decoder(encoded, training=training)
        return decoded


def create_autoencoder_model(input_shape=(256, 256, 3), model_type='denoising'):
    """Factory function to create autoencoder model"""
    if model_type == 'vae':
        model = VariationalAutoencoder(input_shape=input_shape)
    elif model_type == 'anomaly':
        model = AnomalyDetectionAutoencoder(input_shape=input_shape)
    else:  # denoising
        model = DenosingAutoencoder(input_shape=input_shape)
    
    model.build(input_shape=(None, *input_shape))
    return model
