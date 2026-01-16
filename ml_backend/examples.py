"""
Example usage and demonstration scripts
Shows how to use the deepfake detection system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_backend import (
    DeepfakeDetectionTrainer,
    DeepfakeInferenceEngine,
    ImageProcessor,
    VideoProcessor,
    AudioProcessor
)
import json


def example_training():
    """Example: Train deepfake detection models"""
    print("\n" + "="*60)
    print("EXAMPLE: Training Deepfake Detection Models")
    print("="*60)
    
    # Initialize trainer
    trainer = DeepfakeDetectionTrainer()
    
    # Display configuration
    print("\nTraining Configuration:")
    print(json.dumps(trainer.config, indent=2))
    
    # To train, you need data in ./data directory
    # Structure:
    #   data/
    #     real/train/
    #     fake/train/
    #     real/val/
    #     fake/val/
    
    print("\nNote: Ensure training data exists in ./data directory")
    print("Directory structure:")
    print("  data/")
    print("    real/")
    print("      train/  (training real images)")
    print("      val/    (validation real images)")
    print("    fake/")
    print("      train/  (training deepfake images)")
    print("      val/    (validation deepfake images)")
    
    # Uncomment to train (requires data)
    # results = trainer.train_all('./data')
    # print(f"\nTraining Results: {json.dumps(results, indent=2)}")


def example_inference():
    """Example: Use trained models for inference"""
    print("\n" + "="*60)
    print("EXAMPLE: Inference with Trained Models")
    print("="*60)
    
    # Initialize inference engine
    print("\nInitializing inference engine...")
    engine = DeepfakeInferenceEngine('./trained_models')
    
    print(f"Loaded {len(engine.models)} models")
    for model_name in engine.models:
        print(f"  - {model_name}")
    
    # Example image prediction (requires image file)
    example_image = './example_image.jpg'
    if Path(example_image).exists():
        print(f"\nPredicting on {example_image}...")
        result = engine.predict_image(example_image)
        print(json.dumps(result, indent=2))
        print("\nReport:")
        print(engine.get_report(result))
    
    # Example video prediction (requires video file)
    example_video = './example_video.mp4'
    if Path(example_video).exists():
        print(f"\nPredicting on {example_video}...")
        result = engine.predict_video(example_video, num_frames=16)
        print(json.dumps(result, indent=2))
    
    # Example audio prediction (requires audio file)
    example_audio = './example_audio.wav'
    if Path(example_audio).exists():
        print(f"\nPredicting on {example_audio}...")
        result = engine.predict_audio(example_audio)
        print(json.dumps(result, indent=2))


def example_image_processing():
    """Example: Image processing utilities"""
    print("\n" + "="*60)
    print("EXAMPLE: Image Processing")
    print("="*60)
    
    processor = ImageProcessor()
    
    # Load and preprocess image
    example_image = './example_image.jpg'
    if Path(example_image).exists():
        print(f"\nLoading {example_image}...")
        img = processor.load_and_preprocess(example_image, (256, 256))
        print(f"Loaded image shape: {img.shape}")
        print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")
        
        # Extract forensic features
        print("\nExtracting forensic features...")
        features = processor.extract_forensic_features(img)
        print("Forensic Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        # Detect facial landmarks
        print("\nDetecting facial landmarks...")
        landmarks = processor.detect_facial_landmarks(img)
        print("Facial Landmarks:")
        for region, coords in landmarks.items():
            print(f"  {region}: {coords}")
        
        # Data augmentation
        print("\nApplying data augmentation...")
        augmented = processor.augment_image(img, augmentation_factor=0.2)
        print(f"Generated {len(augmented)} augmented versions")


def example_video_processing():
    """Example: Video processing utilities"""
    print("\n" + "="*60)
    print("EXAMPLE: Video Processing")
    print("="*60)
    
    processor = VideoProcessor()
    
    # Extract frames from video
    example_video = './example_video.mp4'
    if Path(example_video).exists():
        print(f"\nExtracting frames from {example_video}...")
        frames = processor.extract_frames(example_video, num_frames=16)
        print(f"Extracted {len(frames)} frames")
        print(f"Frame shape: {frames[0].shape}")
        
        # Detect inconsistencies
        print("\nDetecting facial inconsistencies...")
        inconsistencies = processor.detect_facial_inconsistencies(frames)
        print("Inconsistencies:")
        for key, value in inconsistencies.items():
            print(f"  {key}: {value:.4f}")
        
        # Extract optical flow
        print("\nExtracting optical flow...")
        flow = processor.extract_optical_flow(frames)
        print(f"Optical flow shape: {flow.shape}")
        print(f"Flow variance: {flow.var():.4f}")


def example_audio_processing():
    """Example: Audio processing utilities"""
    print("\n" + "="*60)
    print("EXAMPLE: Audio Processing")
    print("="*60)
    
    processor = AudioProcessor()
    
    # Load and preprocess audio
    example_audio = './example_audio.wav'
    if Path(example_audio).exists():
        print(f"\nLoading {example_audio}...")
        y, sr = processor.load_and_preprocess(example_audio)
        print(f"Audio shape: {y.shape}")
        print(f"Sample rate: {sr}")
        
        # Extract MFCC
        print("\nExtracting MFCC features...")
        mfcc = processor.extract_mfcc(y, sr)
        print(f"MFCC shape: {mfcc.shape}")
        
        # Extract spectrogram
        print("\nExtracting spectrogram...")
        spec = processor.extract_spectrogram(y, sr)
        print(f"Spectrogram shape: {spec.shape}")
        
        # Extract audio features
        print("\nExtracting audio features...")
        features = processor.extract_audio_features(y, sr)
        print("Audio Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        # Detect voice artifacts
        print("\nDetecting voice artifacts...")
        artifacts = processor.detect_voice_artifacts(y, sr)
        print("Voice Artifacts:")
        for key, value in artifacts.items():
            print(f"  {key}: {value:.4f}")


def example_api():
    """Example: Using REST API"""
    print("\n" + "="*60)
    print("EXAMPLE: REST API Usage")
    print("="*60)
    
    print("\nTo use the REST API:")
    print("\n1. Start the API server:")
    print("   python -m ml_backend.inference.api")
    
    print("\n2. Use curl to make predictions:")
    print("\n   Image prediction:")
    print("   curl -X POST \\")
    print("     -F 'file=@image.jpg' \\")
    print("     -F 'confidence_threshold=0.5' \\")
    print("     http://localhost:5000/api/predict/image")
    
    print("\n   Video prediction:")
    print("   curl -X POST \\")
    print("     -F 'file=@video.mp4' \\")
    print("     -F 'num_frames=16' \\")
    print("     http://localhost:5000/api/predict/video")
    
    print("\n   Audio prediction:")
    print("   curl -X POST \\")
    print("     -F 'file=@audio.wav' \\")
    print("     http://localhost:5000/api/predict/audio")
    
    print("\n3. Available endpoints:")
    print("   GET  /api/health           - Health check")
    print("   GET  /api/models           - List models")
    print("   POST /api/predict/image    - Predict image")
    print("   POST /api/predict/video    - Predict video")
    print("   POST /api/predict/audio    - Predict audio")
    print("   POST /api/predict/multimodal - Multimodal prediction")
    print("   POST /api/batch-predict    - Batch prediction")
    print("   GET  /api/stats            - System statistics")


def example_model_configurations():
    """Example: Understanding model configurations"""
    print("\n" + "="*60)
    print("EXAMPLE: Model Configurations")
    print("="*60)
    
    configs = {
        'CNN': {
            'purpose': 'Spatial feature extraction',
            'input': 'Images (256x256x3)',
            'output': 'Binary classification (Real/Fake)',
            'strengths': ['Good at detecting visual artifacts', 'Fast inference'],
            'limitations': ['Limited temporal understanding']
        },
        'RNN/LSTM': {
            'purpose': 'Temporal sequence analysis',
            'input': 'Video frames sequence',
            'output': 'Binary classification',
            'strengths': ['Captures temporal inconsistencies', 'Detects unnatural transitions'],
            'limitations': ['Requires multiple frames', 'Slower inference']
        },
        'Autoencoder': {
            'purpose': 'Anomaly detection',
            'input': 'Images (256x256x3)',
            'output': 'Reconstruction error',
            'strengths': ['Unsupervised learning', 'Good for novel deepfakes'],
            'limitations': ['Harder to interpret']
        },
        'Audio Model': {
            'purpose': 'Voice synthesis detection',
            'input': 'Audio (MFCC, Spectrogram)',
            'output': 'Binary classification',
            'strengths': ['Detects voice artifacts', 'Analyzes speech patterns'],
            'limitations': ['Requires quality audio']
        },
        'Hybrid': {
            'purpose': 'Combined multimodal analysis',
            'input': 'Images + Videos + Audio',
            'output': 'Confidence score',
            'strengths': ['Highest accuracy', 'Robust to variations'],
            'limitations': ['Requires more compute']
        }
    }
    
    for model_name, config in configs.items():
        print(f"\n{model_name}:")
        for key, value in config.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value}")


def example_accuracy_optimization():
    """Example: Strategies for achieving 99% accuracy"""
    print("\n" + "="*60)
    print("EXAMPLE: Achieving 99% Accuracy")
    print("="*60)
    
    strategies = {
        'Data': [
            'Use diverse datasets (real-world, synthetic, lab-created)',
            'Ensure balanced dataset (equal real/fake samples)',
            'Augment training data (rotation, brightness, zoom)',
            'Use high-quality media files'
        ],
        'Model': [
            'Use ensemble methods (combine multiple models)',
            'Apply transfer learning (pre-trained weights)',
            'Use attention mechanisms (focus on critical regions)',
            'Implement custom loss functions (focal loss, weighted BCE)'
        ],
        'Training': [
            'Use learning rate scheduling',
            'Implement early stopping',
            'Apply class weights for imbalance',
            'Use multiple optimizers (Adam, SGD, AdamW)'
        ],
        'Inference': [
            'Combine multiple models (voting/averaging)',
            'Use confidence thresholds appropriately',
            'Implement multimodal analysis',
            'Use ensemble predictions'
        ]
    }
    
    for category, items in strategies.items():
        print(f"\n{category}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION SYSTEM - EXAMPLES AND DEMONSTRATIONS")
    print("="*80)
    
    examples = [
        ("Training", example_training),
        ("Inference", example_inference),
        ("Image Processing", example_image_processing),
        ("Video Processing", example_video_processing),
        ("Audio Processing", example_audio_processing),
        ("REST API", example_api),
        ("Model Configurations", example_model_configurations),
        ("Accuracy Optimization", example_accuracy_optimization)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠ Error in {name}: {e}")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Prepare your training data")
    print("2. Run: python -m ml_backend.training.train")
    print("3. Start API: python -m ml_backend.inference.api")
    print("4. Make predictions via REST API or Python SDK")


if __name__ == "__main__":
    main()
