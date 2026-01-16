"""
Flask API for Deepfake Detection Service
Provides REST endpoints for image, video, and audio analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import json
from datetime import datetime
import threading
from queue import Queue

from inference.engine import get_inference_engine


app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('./uploads')
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'aac', 'flac', 'ogg'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Initialize inference engine
try:
    inference_engine = get_inference_engine('./trained_models')
except Exception as e:
    print(f"Warning: Could not load inference engine: {e}")
    inference_engine = None

# Job queue for async processing
job_queue = Queue()
job_results = {}


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(inference_engine.models) if inference_engine else 0
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of loaded models"""
    if not inference_engine:
        return jsonify({'error': 'Inference engine not loaded'}), 500
    
    models_info = []
    for model_name, model in inference_engine.models.items():
        models_info.append({
            'name': model_name,
            'parameters': model.count_params() if hasattr(model, 'count_params') else 0
        })
    
    return jsonify({
        'models': models_info,
        'total_models': len(models_info)
    })


@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """Predict if uploaded image is deepfake"""
    if not inference_engine:
        return jsonify({'error': 'Inference engine not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': f'Invalid image extension. Allowed: {ALLOWED_IMAGE_EXTENSIONS}'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / f"img_{datetime.now().timestamp()}_{filename}"
        file.save(str(filepath))
        
        # Get confidence threshold
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        
        # Predict
        result = inference_engine.predict_image(str(filepath), confidence_threshold)
        
        # Generate report
        report = inference_engine.get_report(result)
        result['report'] = report
        
        # Cleanup
        try:
            filepath.unlink()
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/video', methods=['POST'])
def predict_video():
    """Predict if uploaded video is deepfake"""
    if not inference_engine:
        return jsonify({'error': 'Inference engine not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': f'Invalid video extension. Allowed: {ALLOWED_VIDEO_EXTENSIONS}'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / f"vid_{datetime.now().timestamp()}_{filename}"
        file.save(str(filepath))
        
        # Get parameters
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        num_frames = int(request.form.get('num_frames', 16))
        
        # Predict
        result = inference_engine.predict_video(
            str(filepath),
            num_frames=num_frames,
            confidence_threshold=confidence_threshold
        )
        
        # Generate report
        report = inference_engine.get_report(result)
        result['report'] = report
        
        # Cleanup
        try:
            filepath.unlink()
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/audio', methods=['POST'])
def predict_audio():
    """Predict if uploaded audio is deepfake"""
    if not inference_engine:
        return jsonify({'error': 'Inference engine not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({'error': f'Invalid audio extension. Allowed: {ALLOWED_AUDIO_EXTENSIONS}'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / f"aud_{datetime.now().timestamp()}_{filename}"
        file.save(str(filepath))
        
        # Get confidence threshold
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        
        # Predict
        result = inference_engine.predict_audio(str(filepath), confidence_threshold)
        
        # Generate report
        report = inference_engine.get_report(result)
        result['report'] = report
        
        # Cleanup
        try:
            filepath.unlink()
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/multimodal', methods=['POST'])
def predict_multimodal():
    """Predict using video and audio"""
    if not inference_engine:
        return jsonify({'error': 'Inference engine not loaded'}), 500
    
    if 'video_file' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video_file']
    audio_file = request.files.get('audio_file', None)
    
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid video extension'}), 400
    
    try:
        # Save video
        filename = secure_filename(video_file.filename)
        video_path = Path(app.config['UPLOAD_FOLDER']) / f"multi_vid_{datetime.now().timestamp()}_{filename}"
        video_file.save(str(video_path))
        
        # Save audio if provided
        audio_path = None
        if audio_file and audio_file.filename != '':
            if allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
                audio_filename = secure_filename(audio_file.filename)
                audio_path = Path(app.config['UPLOAD_FOLDER']) / f"multi_aud_{datetime.now().timestamp()}_{audio_filename}"
                audio_file.save(str(audio_path))
        
        # Get parameters
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        
        # Predict
        result = inference_engine.predict_multimodal(
            str(video_path),
            audio_path=str(audio_path) if audio_path else None,
            confidence_threshold=confidence_threshold
        )
        
        # Generate report
        report = inference_engine.get_report(result)
        result['report'] = report
        
        # Cleanup
        try:
            video_path.unlink()
            if audio_path:
                audio_path.unlink()
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch predict multiple files"""
    if not inference_engine:
        return jsonify({'error': 'Inference engine not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    file_type = request.form.get('file_type', 'image')
    confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        results = []
        file_paths = []
        
        # Save files
        for file in files:
            if file.filename == '':
                continue
            
            if file_type == 'image' and not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                continue
            elif file_type == 'video' and not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
                continue
            elif file_type == 'audio' and not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
                continue
            
            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / f"batch_{datetime.now().timestamp()}_{filename}"
            file.save(str(filepath))
            file_paths.append(str(filepath))
        
        # Predict
        results = inference_engine.batch_predict(file_paths, file_type, confidence_threshold)
        
        # Cleanup
        for filepath in file_paths:
            try:
                Path(filepath).unlink()
            except:
                pass
        
        return jsonify({
            'results': results,
            'total': len(results),
            'deepfakes_detected': sum(1 for r in results if r.get('is_deepfake', False))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(inference_engine.models) if inference_engine else 0,
        'upload_folder': str(UPLOAD_FOLDER),
        'max_file_size': MAX_FILE_SIZE,
        'supported_formats': {
            'images': list(ALLOWED_IMAGE_EXTENSIONS),
            'videos': list(ALLOWED_VIDEO_EXTENSIONS),
            'audio': list(ALLOWED_AUDIO_EXTENSIONS)
        }
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
