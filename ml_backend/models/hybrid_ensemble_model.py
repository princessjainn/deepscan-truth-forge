"""
Hybrid Ensemble Model for Deepfake Detection
Combines CNN, RNN, Autoencoder, and Attention mechanisms
Targets 99% accuracy across multiple modalities
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import EfficientNetB3, DenseNet121, Xception
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, LSTM, Bidirectional, Dense, Dropout,
    Flatten, Concatenate, Input, GlobalAveragePooling2D,
    Attention, MultiHeadAttention, LayerNormalization, Reshape,
    TimeDistributed, Conv3D, MaxPooling3D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import librosa
import scipy.fftpack as fft


class HybridDeepfakeDetector(Model):
    """
    Multi-modal hybrid model combining:
    - 3D CNN for video temporal analysis
    - Multiple 2D CNNs for spatial features
    - RNN/LSTM for sequence modeling
    - Autoencoder for anomaly detection
    - Attention mechanisms for critical region focus
    - Audio analysis for lip-sync verification
    """
    
    def __init__(self, input_shape=(224, 224, 3), include_audio=True):
        super(HybridDeepfakeDetector, self).__init__()
        self.input_shape_val = input_shape
        self.include_audio = include_audio
        
        # ============ Video Processing Branch ============
        
        # 3D CNN for temporal patterns
        self.video_3d_cnn = self._build_3d_cnn()
        
        # Spatial CNN branches (EfficientNet, DenseNet, Xception)
        self.efficientnet = self._build_efficientnet()
        self.densenet = self._build_densenet()
        self.xception = self._build_xception_model()
        
        # ============ Sequence Processing Branch ============
        
        # Bidirectional LSTM for temporal consistency
        self.lstm_branch = self._build_lstm_branch()
        
        # ============ Autoencoder Branch ============
        
        # Reconstruction-based anomaly detection
        self.autoencoder = self._build_autoencoder()
        
        # ============ Audio Processing Branch ============
        
        if self.include_audio:
            self.audio_branch = self._build_audio_branch()
        
        # ============ Attention Mechanisms ============
        
        self.channel_attention = self._build_channel_attention()
        self.spatial_attention = self._build_spatial_attention()
        
        # ============ Fusion and Classification ============
        
        self.fusion_layers = self._build_fusion_network()
        
    def _build_3d_cnn(self):
        """3D CNN for video temporal analysis"""
        model = Sequential([
            Conv3D(32, (3, 3, 3), activation='relu', padding='same', 
                   input_shape=(8, 224, 224, 3), kernel_regularizer=l2(1e-4)),
            MaxPooling3D((2, 2, 2)),
            
            Conv3D(64, (3, 3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(1e-4)),
            MaxPooling3D((2, 2, 2)),
            
            Conv3D(128, (3, 3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(1e-4)),
            MaxPooling3D((2, 2, 2)),
            
            Conv3D(256, (3, 3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(1e-4)),
            GlobalAveragePooling3D(),
            
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.4),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        ], name='3d_cnn')
        return model
    
    def _build_efficientnet(self):
        """Transfer learning with EfficientNetB3"""
        base_model = EfficientNetB3(
            input_shape=self.input_shape_val,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        ], name='efficientnet')
        return model
    
    def _build_densenet(self):
        """Transfer learning with DenseNet121"""
        base_model = DenseNet121(
            input_shape=self.input_shape_val,
            include_top=False,
            weights='imagenet'
        )
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        ], name='densenet')
        return model
    
    def _build_xception_model(self):
        """Transfer learning with Xception"""
        base_model = Xception(
            input_shape=self.input_shape_val,
            include_top=False,
            weights='imagenet'
        )
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        ], name='xception')
        return model
    
    def _build_lstm_branch(self):
        """Bidirectional LSTM for temporal sequence analysis"""
        model = Sequential([
            Bidirectional(LSTM(256, return_sequences=True), 
                         input_shape=(None, 256)),
            Dropout(0.3),
            Bidirectional(LSTM(128, return_sequences=False)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        ], name='lstm_branch')
        return model
    
    def _build_autoencoder(self):
        """Variational Autoencoder for anomaly detection"""
        # Encoder
        inputs = Input(shape=self.input_shape_val)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        
        # Latent space
        latent_dim = 256
        z = Dense(latent_dim, activation='relu')(x)
        
        # Decoder
        x = Dense(128 * 28 * 28, activation='relu')(z)
        x = Reshape((28, 28, 128))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        model = Model(inputs, outputs, name='autoencoder')
        return model
    
    def _build_audio_branch(self):
        """Audio analysis for lip-sync and voice authenticity"""
        model = Sequential([
            Dense(512, activation='relu', input_shape=(256,)),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
        ], name='audio_branch')
        return model
    
    def _build_channel_attention(self):
        """Channel attention mechanism"""
        def channel_attention(x, reduction=16):
            channels = x.shape[-1]
            avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
            
            fc1 = Dense(channels // reduction, activation='relu')
            fc2 = Dense(channels, activation='sigmoid')
            
            avg_out = fc2(fc1(avg_pool))
            max_out = fc2(fc1(max_pool))
            
            return x * (avg_out + max_out)
        
        return channel_attention
    
    def _build_spatial_attention(self):
        """Spatial attention mechanism"""
        def spatial_attention(x):
            avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
            concat = tf.concat([avg_pool, max_pool], axis=-1)
            
            conv = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
            return x * conv
        
        return spatial_attention
    
    def _build_fusion_network(self):
        """Fusion layers combining all branches"""
        model = Sequential([
            Dense(1024, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.4),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ], name='fusion')
        return model
    
    def call(self, inputs, training=False):
        """
        Forward pass through hybrid model
        inputs: dict with 'video', 'frames', 'audio' (optional)
        """
        video_frames = inputs.get('frames')
        audio_features = inputs.get('audio_features', None)
        
        # Process through spatial CNN branches
        efficientnet_out = self.efficientnet(video_frames, training=training)
        densenet_out = self.densenet(video_frames, training=training)
        xception_out = self.xception_model(video_frames, training=training)
        
        # Apply attention
        efficientnet_out = self.channel_attention(
            tf.expand_dims(efficientnet_out, axis=1)
        )
        efficientnet_out = tf.squeeze(efficientnet_out, axis=1)
        
        # 3D CNN processing
        video_3d = self.video_3d_cnn(inputs.get('video_sequence'), training=training)
        
        # Autoencoder reconstruction
        reconstructed = self.autoencoder(video_frames, training=training)
        reconstruction_error = tf.reduce_mean(
            tf.abs(video_frames - reconstructed), axis=[1, 2, 3]
        )
        
        # Concatenate spatial features
        spatial_concat = Concatenate()([
            efficientnet_out, densenet_out, xception_out
        ])
        
        # LSTM processing on concatenated features
        lstm_out = self.lstm_branch(tf.expand_dims(spatial_concat, axis=1), training=training)
        
        # Combine all features
        combined = Concatenate()([
            video_3d, spatial_concat, lstm_out, 
            tf.expand_dims(reconstruction_error, axis=1)
        ])
        
        # Add audio if available
        if self.include_audio and audio_features is not None:
            audio_out = self.audio_branch(audio_features, training=training)
            combined = Concatenate()([combined, audio_out])
        
        # Final classification
        output = self.fusion_layers(combined, training=training)
        
        return output


class AudioDeepfakeDetector(Model):
    """
    Specialized audio deepfake detector for voice authenticity
    and lip-sync verification
    """
    
    def __init__(self, sample_rate=16000):
        super(AudioDeepfakeDetector, self).__init__()
        self.sample_rate = sample_rate
        
        # Spectrogram analysis
        self.spectrogram_cnn = self._build_spectrogram_cnn()
        
        # MFCC analysis
        self.mfcc_lstm = self._build_mfcc_lstm()
        
        # Fusion
        self.fusion = Sequential([
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    
    def _build_spectrogram_cnn(self):
        """CNN for spectrogram analysis"""
        return Sequential([
            Conv1D(64, 5, activation='relu', padding='same'),
            MaxPooling1D(4),
            Conv1D(128, 5, activation='relu', padding='same'),
            MaxPooling1D(4),
            Conv1D(256, 5, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
        ])
    
    def _build_mfcc_lstm(self):
        """LSTM for MFCC coefficient sequences"""
        return Sequential([
            Bidirectional(LSTM(128, return_sequences=True), 
                         input_shape=(None, 13)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dense(128, activation='relu'),
        ])
    
    def call(self, audio_input, training=False):
        """
        Process audio and detect deepfakes
        audio_input: dict with 'waveform', 'spectrogram', 'mfcc'
        """
        spectrogram = audio_input.get('spectrogram')
        mfcc = audio_input.get('mfcc')
        
        spec_features = self.spectrogram_cnn(spectrogram, training=training)
        mfcc_features = self.mfcc_lstm(mfcc, training=training)
        
        combined = Concatenate()([spec_features, mfcc_features])
        output = self.fusion(combined, training=training)
        
        return output


class LipSyncAnalyzer(Model):
    """
    Detects audio-visual lip-sync inconsistencies
    """
    
    def __init__(self):
        super(LipSyncAnalyzer, self).__init__()
        
        self.visual_encoder = Sequential([
            Dense(256, activation='relu', input_shape=(256,)),
            Dropout(0.2),
            Dense(128, activation='relu'),
        ])
        
        self.audio_encoder = Sequential([
            Dense(256, activation='relu', input_shape=(256,)),
            Dropout(0.2),
            Dense(128, activation='relu'),
        ])
        
        self.sync_detector = Sequential([
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs, training=False):
        """
        inputs: dict with 'visual_features', 'audio_features'
        """
        visual = self.visual_encoder(inputs['visual_features'], training=training)
        audio = self.audio_encoder(inputs['audio_features'], training=training)
        
        # Compute correlation
        correlation = tf.reduce_sum(visual * audio, axis=-1, keepdims=True)
        
        combined = Concatenate()([visual, audio, correlation])
        output = self.sync_detector(combined, training=training)
        
        return output


def create_ensemble_model(input_shape=(224, 224, 3), include_audio=True):
    """Factory function to create the hybrid ensemble model"""
    return HybridDeepfakeDetector(input_shape=input_shape, include_audio=include_audio)


if __name__ == "__main__":
    # Test model creation
    model = create_ensemble_model()
    print("Hybrid Deepfake Detection Model created successfully!")
    print(model.summary())
