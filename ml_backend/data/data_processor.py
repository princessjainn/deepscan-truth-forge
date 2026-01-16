"""
Data Preprocessing Pipeline for Multiple Media Types
Handles images, videos, audio, and mixed formats
"""

import cv2
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import tensorflow as tf
from PIL import Image
import json


class ImageProcessor:
    """Process images for deepfake detection"""
    
    @staticmethod
    def load_and_preprocess(image_path, target_size=(256, 256)):
        """Load and preprocess image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        
        return img
    
    @staticmethod
    def detect_facial_landmarks(image):
        """Detect facial landmarks for biometric analysis"""
        # This would use face_recognition or MediaPipe
        # Simplified version for demonstration
        h, w = image.shape[:2]
        landmarks = {
            'face_region': [0, 0, w, h],
            'left_eye': [w//4, h//3, w//4 + 50, h//3 + 50],
            'right_eye': [3*w//4 - 50, h//3, 3*w//4, h//3 + 50],
            'mouth': [2*w//5, 2*h//3, 3*w//5, 2*h//3 + 40]
        }
        return landmarks
    
    @staticmethod
    def extract_forensic_features(image):
        """Extract forensic features for detection"""
        # Compression artifacts
        gray = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        dct = cv2.dct(np.float32(gray) / 255.0)
        dct_energy = np.sum(dct ** 2)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        
        # Color distribution
        hsv = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        features = {
            'dct_energy': float(dct_energy),
            'edge_ratio': float(edge_ratio),
            'h_hist_std': float(np.std(h_hist)),
            's_hist_std': float(np.std(s_hist))
        }
        
        return features
    
    @staticmethod
    def augment_image(image, augmentation_factor=0.2):
        """Data augmentation for training"""
        augmented = []
        augmented.append(image)  # Original
        
        # Brightness adjustment
        brightness = image + np.random.uniform(-augmentation_factor, augmentation_factor)
        augmented.append(np.clip(brightness, 0, 1))
        
        # Slight rotation
        h, w = image.shape[:2]
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine((image * 255).astype('uint8'), M, (w, h))
        augmented.append(rotated.astype('float32') / 255.0)
        
        # Horizontal flip
        flipped = np.fliplr(image)
        augmented.append(flipped)
        
        return augmented


class VideoProcessor:
    """Process videos for deepfake detection"""
    
    @staticmethod
    def extract_frames(video_path, num_frames=16, target_size=(256, 256)):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < num_frames:
            # Pad with last frame if needed
            while len(frames) < num_frames:
                frames.append(frames[-1])
        
        return np.array(frames[:num_frames])
    
    @staticmethod
    def detect_facial_inconsistencies(frames):
        """Detect inconsistencies in facial movement"""
        inconsistencies = {
            'blinking_irregularity': 0,
            'gaze_inconsistency': 0,
            'mouth_sync_error': 0
        }
        
        # Simplified temporal consistency check
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i-1]))
            frame_diffs.append(diff)
        
        # Detect sudden jumps (unnatural transitions)
        frame_diffs = np.array(frame_diffs)
        mean_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)
        
        if std_diff > 0:
            inconsistencies['gaze_inconsistency'] = float(np.sum(frame_diffs > mean_diff + 2*std_diff))
        
        return inconsistencies
    
    @staticmethod
    def extract_optical_flow(frames):
        """Extract optical flow for motion analysis"""
        flow_vectors = []
        gray_frames = []
        
        for frame in frames:
            gray = cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)
        
        for i in range(1, len(gray_frames)):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_vectors.append(mag)
        
        return np.array(flow_vectors)


class AudioProcessor:
    """Process audio for deepfake detection"""
    
    @staticmethod
    def load_and_preprocess(audio_path, sr=22050, duration=5):
        """Load and preprocess audio"""
        y, sr = librosa.load(str(audio_path), sr=sr, duration=duration)
        
        # Normalize
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
        
        return y, sr
    
    @staticmethod
    def extract_mfcc(y, sr=22050, n_mfcc=40):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = librosa.power_to_db(mfcc, ref=np.max)
        
        return mfcc
    
    @staticmethod
    def extract_spectrogram(y, sr=22050, n_fft=2048):
        """Extract mel-spectrogram"""
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
        S = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to 0-1
        S = (S - S.min()) / (S.max() - S.min() + 1e-8)
        
        return S
    
    @staticmethod
    def extract_audio_features(y, sr=22050):
        """Extract various audio features for analysis"""
        features = {}
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spec_centroid_mean'] = float(np.mean(spec_cent))
        features['spec_centroid_std'] = float(np.std(spec_cent))
        
        # Spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spec_rolloff_mean'] = float(np.mean(spec_rolloff))
        features['spec_rolloff_std'] = float(np.std(spec_rolloff))
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfcc))
        features['mfcc_std'] = float(np.std(mfcc))
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        return features
    
    @staticmethod
    def detect_voice_artifacts(y, sr=22050):
        """Detect artifacts suggesting voice synthesis"""
        artifacts = {}
        
        # Harmonic-percussive separation
        D = librosa.stft(y)
        H, P = librosa.decompose.hpss(D)
        
        # Energy ratio
        harmonic_energy = np.sum(np.abs(H)**2)
        percussive_energy = np.sum(np.abs(P)**2)
        total_energy = harmonic_energy + percussive_energy
        
        artifacts['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-8))
        artifacts['percussive_ratio'] = float(percussive_energy / (total_energy + 1e-8))
        
        # Spectral variance (synthesized speech often has unusual variance)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        spectral_variance = np.var(librosa.power_to_db(S, ref=np.max), axis=1)
        artifacts['spectral_variance'] = float(np.mean(spectral_variance))
        
        return artifacts


class DataGenerator:
    """Generate batches of training data"""
    
    def __init__(self, data_dir, batch_size=32, target_size=(256, 256)):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
    
    def load_image_dataset(self, split='train'):
        """Load image dataset"""
        images = []
        labels = []
        
        real_dir = self.data_dir / 'real' / split
        fake_dir = self.data_dir / 'fake' / split
        
        # Load real images
        if real_dir.exists():
            for img_path in real_dir.glob('*.jpg'):
                try:
                    img = self.image_processor.load_and_preprocess(img_path, self.target_size)
                    images.append(img)
                    labels.append(0)  # Real
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Load fake images
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.jpg'):
                try:
                    img = self.image_processor.load_and_preprocess(img_path, self.target_size)
                    images.append(img)
                    labels.append(1)  # Fake
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def load_video_dataset(self, split='train', num_frames=16):
        """Load video dataset"""
        videos = []
        labels = []
        
        real_dir = self.data_dir / 'real_videos' / split
        fake_dir = self.data_dir / 'fake_videos' / split
        
        # Load real videos
        if real_dir.exists():
            for video_path in real_dir.glob('*.mp4'):
                try:
                    frames = self.video_processor.extract_frames(video_path, num_frames, self.target_size)
                    videos.append(frames)
                    labels.append(0)  # Real
                except Exception as e:
                    print(f"Error loading {video_path}: {e}")
        
        # Load fake videos
        if fake_dir.exists():
            for video_path in fake_dir.glob('*.mp4'):
                try:
                    frames = self.video_processor.extract_frames(video_path, num_frames, self.target_size)
                    videos.append(frames)
                    labels.append(1)  # Fake
                except Exception as e:
                    print(f"Error loading {video_path}: {e}")
        
        return np.array(videos), np.array(labels)
    
    def load_audio_dataset(self, split='train'):
        """Load audio dataset"""
        spectrograms = []
        mfccs = []
        labels = []
        
        real_dir = self.data_dir / 'real_audio' / split
        fake_dir = self.data_dir / 'fake_audio' / split
        
        # Load real audio
        if real_dir.exists():
            for audio_path in real_dir.glob('*.wav'):
                try:
                    y, sr = self.audio_processor.load_and_preprocess(audio_path)
                    spec = self.audio_processor.extract_spectrogram(y, sr)
                    mfcc = self.audio_processor.extract_mfcc(y, sr)
                    spectrograms.append(spec)
                    mfccs.append(mfcc)
                    labels.append(0)  # Real
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
        
        # Load fake audio
        if fake_dir.exists():
            for audio_path in fake_dir.glob('*.wav'):
                try:
                    y, sr = self.audio_processor.load_and_preprocess(audio_path)
                    spec = self.audio_processor.extract_spectrogram(y, sr)
                    mfcc = self.audio_processor.extract_mfcc(y, sr)
                    spectrograms.append(spec)
                    mfccs.append(mfcc)
                    labels.append(1)  # Fake
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
        
        return np.array(spectrograms), np.array(mfccs), np.array(labels)


def create_tf_dataset(images, labels, batch_size=32, augment=False):
    """Create TensorFlow dataset"""
    
    def augment_fn(img, label):
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
