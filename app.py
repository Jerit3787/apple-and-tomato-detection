import os
import numpy as np
import datetime
import csv
from io import StringIO, BytesIO
from flask import Flask, request, render_template, jsonify, send_file, Response, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance, UnidentifiedImageError, ImageDraw, ImageFont
import json

# Initialize the Flask application
app = Flask(__name__)

# Global variable to store prediction history
prediction_history = []

# Available models configuration
AVAILABLE_MODELS = {
    'basic': {
        'name': 'Basic CNN',
        'path': 'output/model.keras',
        'class_indices_path': 'output/class_indices.json',
        'img_size': 128,
        'type': 'classification',
        'description': 'Original basic CNN model for classification'
    },
    'resnet50': {
        'name': 'ResNet50 (Advanced)',
        'path': 'output_advanced/best_model_resnet50.keras',
        'class_indices_path': 'output_advanced/class_indices_resnet50.json',
        'img_size': 224,
        'type': 'classification',
        'description': 'Advanced ResNet50 transfer learning model'
    },
    'mobilenetv2': {
        'name': 'MobileNetV2 (Advanced)',
        'path': 'output_advanced/model/best_model_mobilenetv2.keras',
        'class_indices_path': 'output_advanced/class_indices_mobilenetv2.json',
        'img_size': 224,
        'type': 'classification',
        'description': 'Lightweight MobileNetV2 model optimized for speed'
    },
    'vgg16': {
        'name': 'VGG16 (Advanced)',
        'path': 'output_advanced/model/best_model_vgg16 (2).keras',
        'class_indices_path': 'output_advanced/class_indices_vgg16.json',
        'img_size': 224,
        'type': 'classification',
        'description': 'Deep VGG16 transfer learning model'
    },
    'detection': {
        'name': 'Object Detection (ResNet50)',
        'path': 'output_detection/detection_model.keras',
        'class_indices_path': 'output_detection/class_indices.json',  # Use basic indices as fallback
        'img_size': 224,
        'type': 'detection',
        'description': 'Object detection with bounding box localization (86% accuracy)'
    }
}

# Current model configuration
current_model_key = 'basic'  # Default to basic model
current_model = None
current_class_indices = None
current_class_names = None
current_simplified_classes = None
current_indexed_simple_classes = None
current_img_size = None

def load_selected_model(model_key):
    """Load the selected model and its configuration"""
    global current_model, current_class_indices, current_class_names
    global current_simplified_classes, current_indexed_simple_classes, current_img_size
    global current_model_key
    
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_config = AVAILABLE_MODELS[model_key]
    
    # Load the model
    print(f"Loading model: {model_config['name']}")
    current_model = load_model(model_config['path'])
    current_img_size = model_config['img_size']
    current_model_key = model_key
    
    # Try to load class indices, fallback to basic model's indices if not found
    try:
        with open(model_config['class_indices_path'], "r") as f:
            current_class_indices = json.load(f)
    except FileNotFoundError:
        print(f"Class indices not found for {model_key}, using basic model indices")
        with open(AVAILABLE_MODELS['basic']['class_indices_path'], "r") as f:
            current_class_indices = json.load(f)
    
    # Define the classes
    current_class_names = list(current_class_indices.keys())
    
    # Create a simplified mapping for prediction display
    current_simplified_classes = {}
    for name in current_class_names:
        if 'apple' in name.lower() or 'Apple' in name:
            current_simplified_classes[name] = 'apple'
        elif 'tomato' in name.lower() or 'Tomato' in name:
            current_simplified_classes[name] = 'tomato'
        else:
            current_simplified_classes[name] = name
    
    print("Original classes:", current_class_names)
    print("Simplified to:", set(current_simplified_classes.values()))
    
    # Create a mapping that preserves the original indices
    current_indexed_simple_classes = {}
    for name, idx in current_class_indices.items():
        current_indexed_simple_classes[idx] = current_simplified_classes[name]
    
    # Create the final class_names array that maintains the original positions
    current_class_names = [current_indexed_simple_classes[i] for i in range(len(current_class_indices))]
    
    print(f"Final class names for display (preserving indices): {current_class_names}")
    print(f"Model loaded successfully: {model_config['name']}")

# Load default model
load_selected_model('basic')

# Legacy variables for backward compatibility
model = current_model
class_indices = current_class_indices
class_names = current_class_names
simplified_classes = current_simplified_classes
indexed_simple_classes = current_indexed_simple_classes
IMG_SIZE = current_img_size

# Define the upload folder
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Statistics dictionary to track usage
stats = {
    'total_predictions': 0,
    'class_counts': {},
    'average_confidence': 0
}

def is_supported_image_format(filename):
    """
    Check if the given filename has a supported image format extension.
    Returns (is_supported, message)
    """
    supported_formats = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'
    }
    
    # Get the file extension and convert to lowercase
    ext = os.path.splitext(filename.lower())[1]
    
    if ext in supported_formats:
        return True, None
    
    # Handle unsupported formats with specific messages
    if ext == '.avif':
        return False, "AVIF image format is not supported. Please convert to JPEG or PNG."
    elif ext == '.heic' or ext == '.heif':
        return False, "HEIC/HEIF image format is not supported. Please convert to JPEG or PNG."
    elif ext == '.svg':
        return False, "SVG vector format is not supported. Please convert to a raster format like PNG."
    elif ext == '.raw' or ext in {'.cr2', '.nef', '.arw'}:
        return False, "Camera RAW formats are not supported. Please convert to JPEG or PNG."
    else:
        return False, f"Unsupported image format: {ext}. Please use JPEG, PNG, GIF, WebP, or TIFF."

def safe_load_image(file_path, target_size=None):
    """
    Safely load an image file of any format including AVIF, WebP, etc.
    Returns preprocessed image array ready for model prediction and the PIL image object.
    """
    try:
        # Open the image using PIL with AVIF support
        pil_img = Image.open(file_path)
        
        # Convert to RGB mode if not already (handles RGBA, grayscale, etc.)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        if target_size:
            pil_img = pil_img.resize(target_size)
            
        # Convert to numpy array
        img_array = np.array(pil_img) / 255.0
        
        # Add batch dimension if needed
        if len(img_array.shape) == 3:  # If image is already 3D (height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)
            
        return img_array, pil_img
    except UnidentifiedImageError as e:
        print(f"Error loading image {file_path}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading image {file_path}: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html', 
                          history=prediction_history[:5],
                          current_model=AVAILABLE_MODELS[current_model_key]['name'],
                          available_models=AVAILABLE_MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part", history=prediction_history[:5])
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction="No selected file", history=prediction_history[:5])
    
    # Check if the file format is supported
    is_supported, message = is_supported_image_format(file.filename)
    if not is_supported:
        return render_template('index.html', 
                              error=message,
                              history=prediction_history[:5])
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Apply enhancement if specified
    enhancement_type = request.form.get('enhancement', 'none')
    if enhancement_type != 'none':
        factor = float(request.form.get('factor', 1.5))
        file_path = enhance_image_file(file_path, enhancement_type, factor)

    try:
        # Load and preprocess the image using our safe image loader
        img_array, _ = safe_load_image(file_path, target_size=(current_img_size, current_img_size))

        # Make prediction with detection awareness
        model_type = AVAILABLE_MODELS[current_model_key]['type']
        predicted_idx, predicted_class, confidence, predictions, bbox = predict_with_model(img_array, model_type)
        
        # Apply confidence threshold
        confidence_message = f"{confidence:.1%}"
        
        # Get original class name for debugging
        original_class_name = list(current_class_indices.keys())[list(current_class_indices.values()).index(predicted_idx)]
        print(f"Prediction: {original_class_name} ({predicted_class}) with confidence: {confidence:.4f}")
        
        # Add bounding box info if detection model
        bbox_info = None
        if bbox is not None:
            bbox_info = bbox
            print(f"Bounding box: {bbox}")
        
        # Update statistics
        stats['total_predictions'] += 1
        if predicted_class in stats['class_counts']:
            stats['class_counts'][predicted_class] += 1
        else:
            stats['class_counts'][predicted_class] = 1
        
        # Update running average confidence
        stats['average_confidence'] = ((stats['average_confidence'] * (stats['total_predictions'] - 1)) + confidence) / stats['total_predictions']
        
        # Create history entry
        history_entry = {
            'id': len(prediction_history),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': file.filename,
            'file_path': file_path,
            'prediction': predicted_class,
            'confidence': confidence,
            'confidence_message': confidence_message,
            'original_class': original_class_name,
            'model_type': model_type,
            'bbox': bbox_info
        }
        
        # Add to history (limit to most recent 20)
        prediction_history.insert(0, history_entry)
        if len(prediction_history) > 20:
            prediction_history.pop()

        # Format image URL correctly for web access
        image_filename = os.path.basename(file_path)
        image_url = f"/uploads/{image_filename}"

        return render_template('index.html', 
                            prediction=predicted_class, 
                            confidence=confidence_message,
                            number=predicted_idx,
                            prediction_id=history_entry['id'],  # Add the correct history entry ID
                            original_class=original_class_name,
                            image_path=image_url,
                            bbox=bbox_info,
                            model_type=model_type,
                            current_model=AVAILABLE_MODELS[current_model_key]['name'],
                            history=prediction_history[:5])
    except Exception as e:
        print(f"Error processing image: {e}")
        return render_template('index.html', 
                             error=str(e),
                             history=prediction_history[:5])

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, 'temp_capture.jpg')
    file.save(file_path)

    # Load and preprocess the image
    img_array, _ = safe_load_image(file_path, target_size=(current_img_size, current_img_size))

    # Make prediction with detection awareness
    model_type = AVAILABLE_MODELS[current_model_key]['type']
    predicted_idx, predicted_class, confidence, predictions, bbox = predict_with_model(img_array, model_type)
    
    # Get detailed class information
    original_class_name = list(current_class_indices.keys())[list(current_class_indices.values()).index(predicted_idx)]
    
    # Apply confidence threshold for better reliability
    if confidence < 0.5:
        prediction_message = f"Uncertain, but looks like a {predicted_class}"
    else:
        prediction_message = predicted_class
    
    # Log prediction details for debugging
    print(f"Prediction: {original_class_name} ({predicted_class}) with confidence: {confidence:.4f}")
    if bbox:
        print(f"Bounding box: {bbox}")
    
    # Update statistics
    stats['total_predictions'] += 1
    if predicted_class in stats['class_counts']:
        stats['class_counts'][predicted_class] += 1
    else:
        stats['class_counts'][predicted_class] = 1
    
    # Update running average confidence
    stats['average_confidence'] = ((stats['average_confidence'] * (stats['total_predictions'] - 1)) + confidence) / stats['total_predictions']
    
    # Create history entry for camera capture
    history_entry = {
        'id': len(prediction_history),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': 'camera_capture.jpg',
        'file_path': file_path,
        'prediction': predicted_class,
        'confidence': confidence,
        'confidence_message': f"{confidence:.1%}",
        'original_class': original_class_name,
        'model_type': model_type,
        'bbox': bbox
    }
    
    # Add to history
    prediction_history.insert(0, history_entry)
    if len(prediction_history) > 20:
        prediction_history.pop()
    
    # Prepare response
    response_data = {
        'prediction': prediction_message,
        'confidence': confidence,
        'confidence_percent': f"{confidence:.1%}",
        'class_index': int(predicted_idx),
        'original_class': original_class_name,
        'model_type': model_type,
        'all_confidences': {current_class_names[i]: float(predictions[0][i]) for i in range(len(current_class_names)) 
                           if predictions[0][i] > 0.01}  # Only return significant confidences
    }
    
    # Add bounding box info if detection model
    if bbox:
        response_data['bbox'] = bbox
    
    return jsonify(response_data)

# New routes for model management

@app.route('/models')
def model_dashboard():
    """Dashboard to view and switch between available models"""
    return render_template('models.html', 
                          available_models=AVAILABLE_MODELS,
                          current_model=current_model_key)

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    new_model_key = request.form.get('model_key')
    
    if not new_model_key or new_model_key not in AVAILABLE_MODELS:
        return jsonify({'error': 'Invalid model selection'})
    
    try:
        # Load the new model
        load_selected_model(new_model_key)
        
        # Update legacy variables for backward compatibility
        global model, class_indices, class_names, simplified_classes, indexed_simple_classes, IMG_SIZE
        model = current_model
        class_indices = current_class_indices
        class_names = current_class_names
        simplified_classes = current_simplified_classes
        indexed_simple_classes = current_indexed_simple_classes
        IMG_SIZE = current_img_size
        
        return jsonify({
            'success': True, 
            'message': f'Successfully switched to {AVAILABLE_MODELS[new_model_key]["name"]}',
            'model_name': AVAILABLE_MODELS[new_model_key]["name"],
            'model_key': new_model_key
        })
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'})

@app.route('/model_info')
def model_info():
    """Get current model information"""
    return jsonify({
        'current_model': current_model_key,
        'model_name': AVAILABLE_MODELS[current_model_key]['name'],
        'description': AVAILABLE_MODELS[current_model_key]['description'],
        'img_size': current_img_size,
        'available_models': {k: {'name': v['name'], 'description': v['description']} 
                           for k, v in AVAILABLE_MODELS.items()}
    })

# New routes for enhanced features

@app.route('/history')
def view_history():
    return render_template('history.html', 
                          history=prediction_history,
                          current_model=AVAILABLE_MODELS[current_model_key]['name'])

@app.route('/analysis/<int:prediction_id>')
def analysis(prediction_id):
    # Find the prediction in history
    prediction_entry = None
    for entry in prediction_history:
        if entry['id'] == prediction_id:
            prediction_entry = entry
            break
    
    if not prediction_entry:
        print(f"Error: Prediction ID {prediction_id} not found in history")
        return redirect(url_for('home'))
    
    file_path = prediction_entry['file_path']
    print(f"Analysis for ID {prediction_id}: Found prediction entry for {prediction_entry['filename']}")
    
    try:
        # Load and preprocess the image using safe_load_image
        img_array, _ = safe_load_image(file_path, target_size=(current_img_size, current_img_size))
        
        # Get model type from prediction entry
        model_type = prediction_entry.get('model_type', 'classification')
        
        # Handle different model types for prediction
        if model_type == 'detection':
            # For detection models, we need to handle different prediction structure
            predictions = current_model.predict(img_array)
            
            # Extract classification probabilities (first part of output)
            classification_probs = predictions[0] if isinstance(predictions, list) else predictions
            
            # Get all predictions sorted by confidence
            all_predictions = []
            for i, prob in enumerate(classification_probs[0]):
                if prob > 0.001:  # Only include non-zero probabilities
                    class_name = list(current_class_indices.keys())[list(current_class_indices.values()).index(i)]
                    simplified = current_simplified_classes[class_name]
                    all_predictions.append({
                        'index': i,
                        'class_name': class_name,
                        'simple_class': simplified,
                        'confidence': float(prob),
                        'confidence_percent': f"{prob:.2%}"
                    })
            
            # Sort by confidence (highest first)
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Get top 5 predictions
            top_predictions = all_predictions[:5]
        else:
            # Standard classification model
            predictions = current_model.predict(img_array)
            
            # Get all predictions sorted by confidence
            all_predictions = []
            for i, prob in enumerate(predictions[0]):
                if prob > 0.001:  # Only include non-zero probabilities
                    class_name = list(current_class_indices.keys())[list(current_class_indices.values()).index(i)]
                    simplified = current_simplified_classes[class_name]
                    all_predictions.append({
                        'index': i,
                        'class_name': class_name,
                        'simple_class': simplified,
                        'confidence': float(prob),
                        'confidence_percent': f"{prob:.2%}"
                    })
            
            # Sort by confidence (highest first)
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Get top 5 predictions
            top_predictions = all_predictions[:5]
        
        # Format the image path correctly for URL access
        image_filename = os.path.basename(file_path)
        image_url = f"/uploads/{image_filename}"
        
        # If it's a detection model with bbox, use the bbox image instead
        if model_type == 'detection' and prediction_entry.get('bbox'):
            image_url = f"/draw_bbox/{prediction_id}"
        
        return render_template('analysis.html', 
                              prediction=prediction_entry,
                              top_predictions=top_predictions,
                              all_predictions=all_predictions,
                              image_path=image_url)
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('home'))

@app.route('/batch', methods=['GET', 'POST'])
def batch_process():
    if request.method == 'POST':
        batch_results = []
        batch_errors = []
        
        if 'files' not in request.files:
            return render_template('batch.html', error="No files part")
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return render_template('batch.html', error="No selected files")
        
        for file in files:
            if file.filename == '':
                continue
            
            # Check if the file format is supported
            is_supported, message = is_supported_image_format(file.filename)
            if not is_supported:
                batch_errors.append({
                    'filename': file.filename,
                    'error': message
                })
                continue
                
            # Process each file similar to single predict
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            try:
                # Process image and make prediction
                img_array, _ = safe_load_image(file_path, target_size=(IMG_SIZE, IMG_SIZE))
                
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                predicted_class = class_names[predicted_idx]
                confidence = float(np.max(predictions[0]))
                original_class_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_idx)]
                
                # Add to history
                history_entry = {
                    'id': len(prediction_history),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'filename': file.filename,
                    'file_path': file_path,
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'confidence_message': f"{confidence:.1%}",
                    'original_class': original_class_name
                }
                
                prediction_history.insert(0, history_entry)
                if len(prediction_history) > 20:
                    prediction_history.pop()
                
                # Format image URL correctly for web access
                image_filename = os.path.basename(file_path)
                image_url = f"/uploads/{image_filename}"
                
                batch_results.append({
                    'id': history_entry['id'],
                    'filename': file.filename,
                    'prediction': predicted_class,
                    'confidence': f"{confidence:.1%}",
                    'file_path': image_url
                })
                
                # Update statistics
                stats['total_predictions'] += 1
                if predicted_class in stats['class_counts']:
                    stats['class_counts'][predicted_class] += 1
                else:
                    stats['class_counts'][predicted_class] = 1
                
                # Update running average confidence
                stats['average_confidence'] = ((stats['average_confidence'] * (stats['total_predictions'] - 1)) + confidence) / stats['total_predictions']
            
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                # Add to error list
                batch_errors.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return render_template('batch_results.html', results=batch_results, errors=batch_errors)
    
    return render_template('batch.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare_images():
    if request.method == 'POST':
        # Get the two selected images IDs from form
        id1 = int(request.form.get('image1', -1))
        id2 = int(request.form.get('image2', -1))
        
        # Find both predictions
        prediction1 = None
        prediction2 = None
        
        for entry in prediction_history:
            if entry['id'] == id1:
                prediction1 = entry
            if entry['id'] == id2:
                prediction2 = entry
                
            # Break if we've found both
            if prediction1 and prediction2:
                break
        
        # If we found both predictions, render comparison
        if prediction1 and prediction2:
            # Format image URLs correctly for web access
            image1_filename = os.path.basename(prediction1['file_path'])
            image2_filename = os.path.basename(prediction2['file_path'])
            image1_url = f"/uploads/{image1_filename}"
            image2_url = f"/uploads/{image2_filename}"
            
            return render_template('comparison.html', 
                                  prediction1=prediction1, 
                                  prediction2=prediction2,
                                  image1_path=image1_url,
                                  image2_path=image2_url,
                                  confidence1=prediction1['confidence_message'],
                                  confidence2=prediction2['confidence_message'],
                                  history=prediction_history)
    
    # Default - show form with history
    return render_template('compare.html', history=prediction_history)

@app.route('/enhance', methods=['POST'])
def enhance_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    enhancement_type = request.form.get('enhancement', 'none')
    factor = float(request.form.get('factor', 1.5))
    
    # Save original
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Apply enhancement
    enhanced_path = enhance_image_file(file_path, enhancement_type, factor)
    
    return jsonify({'enhanced_path': enhanced_path.replace('\\', '/')})

def enhance_image_file(file_path, enhancement_type, factor=1.5):
    # Open with PIL
    pil_image = Image.open(file_path)
    
    # Apply enhancement
    if enhancement_type == 'brightness':
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
    elif enhancement_type == 'contrast':
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
    elif enhancement_type == 'sharpness':
        enhancer = ImageEnhance.Sharpness(pil_image)
        enhanced = enhancer.enhance(factor)
    elif enhancement_type == 'color':
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(factor)
    else:
        # No enhancement
        return file_path
    
    # Save enhanced
    base_name = os.path.basename(file_path)
    enhanced_path = os.path.join(UPLOAD_FOLDER, f"enhanced_{enhancement_type}_{base_name}")
    enhanced.save(enhanced_path)
    
    return enhanced_path

@app.route('/stats')
def statistics_dashboard():
    # Get counts for apple vs tomato
    apple_count = stats['class_counts'].get('apple', 0)
    tomato_count = stats['class_counts'].get('tomato', 0)
    
    # Calculate average confidence
    avg_confidence = stats['average_confidence'] if stats['total_predictions'] > 0 else 0
    
    return render_template('stats.html', 
                          stats=stats, 
                          apple_count=apple_count,
                          tomato_count=tomato_count,
                          avg_confidence=avg_confidence)

@app.route('/export/csv')
def export_csv():
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Timestamp', 'Filename', 'Prediction', 'Original Class', 'Confidence'])
    
    # Write rows
    for entry in prediction_history:
        writer.writerow([
            entry['id'],
            entry['timestamp'],
            entry['filename'],
            entry['prediction'],
            entry['original_class'],
            f"{entry['confidence']:.4f}"
        ])
    
    # Create response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=fruit_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    prediction_id = request.form.get('prediction_id')
    actual_class = request.form.get('actual_class')
    
    # Find the prediction in history
    for entry in prediction_history:
        if entry['id'] == int(prediction_id):
            predicted_class = entry['prediction']
            
            # Store feedback in file
            feedback_file = 'feedback_log.csv'
            file_exists = os.path.isfile(feedback_file)
            
            with open(feedback_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Prediction ID', 'Filename', 'Predicted Class', 'Actual Class', 'Correct'])
                
                writer.writerow([
                    datetime.datetime.now(), 
                    prediction_id, 
                    entry['filename'],
                    predicted_class,
                    actual_class,
                    predicted_class == actual_class
                ])
            
            return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})
    
    return jsonify({'status': 'error', 'message': 'Prediction not found'})

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/requirements.txt')
def serve_requirements():
    """Endpoint to download requirements.txt file"""
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-disposition": "attachment; filename=requirements.txt"}
    )

def predict_with_model(img_array, model_type='classification'):
    """Make prediction with current model, handling both classification and detection"""
    predictions = current_model.predict(img_array)
    
    if model_type == 'detection':
        # Object detection model outputs: [classification_probs, bbox_coords]
        # predictions[0] = classification probabilities
        # predictions[1] = bounding box coordinates [x, y, width, height]
        classification_probs = predictions[0] if isinstance(predictions, list) else predictions
        bbox_coords = predictions[1] if isinstance(predictions, list) and len(predictions) > 1 else None
        
        predicted_idx = np.argmax(classification_probs[0])
        predicted_class = current_class_names[predicted_idx]
        confidence = float(np.max(classification_probs[0]))
        
        # Get bounding box if available
        bbox = None
        if bbox_coords is not None:
            bbox = {
                'x': float(bbox_coords[0][0]),
                'y': float(bbox_coords[0][1]), 
                'width': float(bbox_coords[0][2]),
                'height': float(bbox_coords[0][3])
            }
        
        return predicted_idx, predicted_class, confidence, classification_probs, bbox
    else:
        # Standard classification model
        predicted_idx = np.argmax(predictions[0])
        predicted_class = current_class_names[predicted_idx]
        confidence = float(np.max(predictions[0]))
        
        return predicted_idx, predicted_class, confidence, predictions, None

@app.route('/draw_bbox/<int:prediction_id>')
def draw_bbox_image(prediction_id):
    """Generate an image with bounding box drawn on it"""
    # Find the prediction in history
    prediction_entry = None
    for entry in prediction_history:
        if entry['id'] == prediction_id:
            prediction_entry = entry
            break
    
    if not prediction_entry or not prediction_entry.get('bbox'):
        # If no bbox data, just serve the original image
        return redirect(url_for('serve_upload', filename=os.path.basename(prediction_entry['file_path'])))
    
    try:
        # Open the original image
        file_path = prediction_entry['file_path']
        image = Image.open(file_path)
        draw = ImageDraw.Draw(image)
        
        # Get bounding box
        bbox = prediction_entry['bbox']
        img_width, img_height = image.size
        
        # Convert normalized coordinates to pixel coordinates
        x = bbox['x'] * img_width
        y = bbox['y'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height
        
        # Draw the bounding box with a 3px red line
        draw.rectangle([(x, y), (x + width, y + height)], outline='red', width=3)
        
        # Draw fruit label
        font_size = int(img_height / 25)  # Proportional font size
        try:
            # Try to load a font, fallback to default if not available
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        label = prediction_entry['prediction'].upper() + f" ({prediction_entry['confidence']:.1%})"
        text_width, text_height = draw.textsize(label, font=font) if hasattr(draw, 'textsize') else (font_size * len(label) * 0.6, font_size * 1.2)
        
        # Draw background for text
        draw.rectangle([(x, y - text_height - 4), (x + text_width + 4, y)], fill='red')
        # Draw text
        draw.text((x + 2, y - text_height - 2), label, fill='white', font=font)
        
        # Serve the image
        img_io = BytesIO()
        image.save(img_io, format='JPEG', quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"Error generating bbox image: {str(e)}")
        # If something goes wrong, just serve the original image
        return redirect(url_for('serve_upload', filename=os.path.basename(prediction_entry['file_path'])))