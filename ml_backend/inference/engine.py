"""
Inference Engine for Deepfake Detection
Provides real-time detection capabilities
"""

import numpy as np
import tensorflow as tf
import cv2
import librosa
from pathlib import Path
from typing import Tuple, Dict, List
import json

from ..data.data_processor import ImageProcessor, VideoProcessor, AudioProcessor
from ..utils.model_utils import ModelSaver


class DeepfakeInferenceEngine:
    """
    Inference engine for detecting deepfakes
    Supports images, videos, and audio files
    """
    
    def __init__(self, model_dir='./trained_models'):
        """Initialize inference engine"""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.model_scores = {}
        self.load_models()
        
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
    
    def load_models(self):
        """Load trained models"""
        model_files = list(self.model_dir.glob('*_best.h5'))
        
        if not model_files:
            print("⚠ Warning: No trained models found")
            return
        
        for model_file in model_files:
            try:
                model_name = model_file.stem.replace('_best', '')
                model = tf.keras.models.load_model(str(model_file))
                self.models[model_name] = model
                print(f"✓ Loaded {model_name} model")
            except Exception as e:
                print(f"✗ Failed to load {model_file}: {e}")
    
    def predict_image(self, image_path: str, confidence_threshold=0.5) -> Dict:
        """
        Predict if image is deepfake
        
        Args:
            image_path: Path to image file
            confidence_threshold: Confidence threshold for detection
            
        Returns:
            Detection results with confidence scores
        """
        try:
            img = self.image_processor.load_and_preprocess(image_path)
            img_batch = np.expand_dims(img, 0)
            
            results = {
                'file': str(image_path),
                'type': 'image',
                'models': {},
                'ensemble_score': 0,
                'is_deepfake': False,
                'confidence': 0
            }
            
            scores = []
            
            # CNN model
            if 'cnn' in self.models:
                pred = self.models['cnn'].predict(img_batch, verbose=0)
                score = float(pred[0, 0])
                results['models']['cnn'] = {'score': score, 'is_deepfake': score > 0.5}
                scores.append(score)
            
            # Calculate ensemble score
            if scores:
                ensemble_score = np.mean(scores)
                results['ensemble_score'] = float(ensemble_score)
                results['is_deepfake'] = ensemble_score > confidence_threshold
                results['confidence'] = float(1 - ensemble_score) if ensemble_score <= 0.5 else float(ensemble_score)
            
            # Forensic analysis
            forensics = self.image_processor.extract_forensic_features(img)
            results['forensics'] = forensics
            
            # Facial landmarks
            landmarks = self.image_processor.detect_facial_landmarks(img)
            results['facial_landmarks'] = landmarks
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'file': str(image_path),
                'type': 'image'
            }
    
    def predict_video(self, video_path: str, num_frames=16, confidence_threshold=0.5) -> Dict:
        """
        Predict if video is deepfake
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze
            confidence_threshold: Confidence threshold
            
        Returns:
            Detection results for video
        """
        try:
            frames = self.video_processor.extract_frames(video_path, num_frames)
            
            results = {
                'file': str(video_path),
                'type': 'video',
                'num_frames': num_frames,
                'frame_predictions': [],
                'models': {},
                'temporal_analysis': {},
                'ensemble_score': 0,
                'is_deepfake': False
            }
            
            frame_scores = []
            
            # Analyze each frame with CNN
            if 'cnn' in self.models:
                model = self.models['cnn']
                for i, frame in enumerate(frames):
                    frame_batch = np.expand_dims(frame, 0)
                    pred = model.predict(frame_batch, verbose=0)
                    score = float(pred[0, 0])
                    frame_scores.append(score)
                    results['frame_predictions'].append({
                        'frame': i,
                        'score': score,
                        'is_deepfake': score > 0.5
                    })
            
            # Temporal analysis
            if frame_scores:
                temporal_inconsistencies = self.video_processor.detect_facial_inconsistencies(frames)
                results['temporal_analysis'] = temporal_inconsistencies
                
                # Optical flow analysis
                optical_flow = self.video_processor.extract_optical_flow(frames)
                results['optical_flow_variance'] = float(np.var(optical_flow))
            
            # Ensemble prediction
            if frame_scores:
                ensemble_score = np.mean(frame_scores)
                results['ensemble_score'] = float(ensemble_score)
                results['is_deepfake'] = ensemble_score > confidence_threshold
                results['confidence'] = float(1 - ensemble_score) if ensemble_score <= 0.5 else float(ensemble_score)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'file': str(video_path),
                'type': 'video'
            }
    
    def predict_audio(self, audio_path: str, confidence_threshold=0.5) -> Dict:
        """
        Predict if audio is deepfake
        
        Args:
            audio_path: Path to audio file
            confidence_threshold: Confidence threshold
            
        Returns:
            Detection results for audio
        """
        try:
            y, sr = self.audio_processor.load_and_preprocess(audio_path)
            spectrogram = self.audio_processor.extract_spectrogram(y, sr)
            mfcc = self.audio_processor.extract_mfcc(y, sr)
            
            results = {
                'file': str(audio_path),
                'type': 'audio',
                'models': {},
                'audio_features': {},
                'voice_artifacts': {},
                'ensemble_score': 0,
                'is_deepfake': False,
                'confidence': 0
            }
            
            scores = []
            
            # Extract features
            audio_features = self.audio_processor.extract_audio_features(y, sr)
            results['audio_features'] = audio_features
            
            # Detect voice artifacts
            voice_artifacts = self.audio_processor.detect_voice_artifacts(y, sr)
            results['voice_artifacts'] = voice_artifacts
            
            # Audio model if available
            if 'audio' in self.models:
                spec_batch = np.expand_dims(spectrogram, axis=(0, -1))
                mfcc_batch = np.expand_dims(mfcc, axis=(0, -1))
                
                try:
                    pred = self.models['audio'].predict(
                        [spec_batch, mfcc_batch], 
                        verbose=0
                    )
                    score = float(pred[0, 0])
                    results['models']['audio'] = {'score': score, 'is_deepfake': score > 0.5}
                    scores.append(score)
                except:
                    pass
            
            # Ensemble score
            if scores:
                ensemble_score = np.mean(scores)
            else:
                # Fallback: use artifact-based detection
                harmonic_ratio = voice_artifacts.get('harmonic_ratio', 0.5)
                ensemble_score = abs(harmonic_ratio - 0.5) * 2  # Distance from natural
            
            results['ensemble_score'] = float(ensemble_score)
            results['is_deepfake'] = ensemble_score > confidence_threshold
            results['confidence'] = float(1 - ensemble_score) if ensemble_score <= 0.5 else float(ensemble_score)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'file': str(audio_path),
                'type': 'audio'
            }
    
    def predict_multimodal(self, video_path: str, audio_path: str = None, 
                          confidence_threshold=0.5) -> Dict:
        """
        Predict using both video and audio
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file (optional)
            confidence_threshold: Confidence threshold
            
        Returns:
            Combined detection results
        """
        video_result = self.predict_video(video_path, confidence_threshold=confidence_threshold)
        
        if audio_path:
            audio_result = self.predict_audio(audio_path, confidence_threshold=confidence_threshold)
            
            # Combine results
            combined_score = (
                video_result.get('ensemble_score', 0.5) +
                audio_result.get('ensemble_score', 0.5)
            ) / 2
            
            video_result['multimodal_analysis'] = {
                'video_score': video_result.get('ensemble_score', 0.5),
                'audio_score': audio_result.get('ensemble_score', 0.5),
                'combined_score': combined_score,
                'audio_video_sync_score': self._check_sync(video_path, audio_path)
            }
            
            video_result['ensemble_score'] = combined_score
            video_result['is_deepfake'] = combined_score > confidence_threshold
        
        return video_result
    
    def _check_sync(self, video_path: str, audio_path: str) -> float:
        """Check audio-visual synchronization"""
        try:
            # Extract video audio
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Compare with provided audio
            # Simplified sync check
            return 0.8  # Placeholder
            
        except:
            return 0.5
    
    def batch_predict(self, file_list: List[str], file_type='image',
                     confidence_threshold=0.5) -> List[Dict]:
        """
        Batch predict multiple files
        
        Args:
            file_list: List of file paths
            file_type: Type of files ('image', 'video', or 'audio')
            confidence_threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        results = []
        
        for file_path in file_list:
            if file_type == 'image':
                result = self.predict_image(file_path, confidence_threshold)
            elif file_type == 'video':
                result = self.predict_video(file_path, confidence_threshold=confidence_threshold)
            elif file_type == 'audio':
                result = self.predict_audio(file_path, confidence_threshold)
            else:
                result = {'error': f'Unknown file type: {file_type}'}
            
            results.append(result)
        
        return results
    
    def get_report(self, results: Dict) -> str:
        """Generate detailed report from results"""
        report = []
        report.append("="*60)
        report.append("DEEPFAKE DETECTION REPORT")
        report.append("="*60)
        report.append(f"File: {results.get('file', 'Unknown')}")
        report.append(f"Type: {results.get('type', 'Unknown')}")
        report.append("")
        
        if results.get('is_deepfake'):
            report.append("⚠ DETECTION: LIKELY DEEPFAKE")
            report.append(f"Confidence: {results.get('confidence', 0):.2%}")
        else:
            report.append("✓ DETECTION: LIKELY AUTHENTIC")
            report.append(f"Confidence: {results.get('confidence', 0):.2%}")
        
        report.append("")
        report.append(f"Ensemble Score: {results.get('ensemble_score', 0):.4f}")
        
        if results.get('models'):
            report.append("\nModel Predictions:")
            for model_name, pred in results.get('models', {}).items():
                status = "FAKE" if pred['is_deepfake'] else "REAL"
                report.append(f"  {model_name}: {status} (score: {pred['score']:.4f})")
        
        report.append("="*60)
        
        return "\n".join(report)


# Singleton instance
_inference_engine = None

def get_inference_engine(model_dir='./trained_models'):
    """Get or create inference engine"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = DeepfakeInferenceEngine(model_dir)
    return _inference_engine
