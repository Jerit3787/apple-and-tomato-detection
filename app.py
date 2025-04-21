import os
import numpy as np
import datetime
import csv
from io import StringIO, BytesIO
from flask import Flask, request, render_template, jsonify, send_file, Response, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance, UnidentifiedImageError
import json

# Initialize the Flask application
app = Flask(__name__)

# Global variable to store prediction history
prediction_history = []

# Load the trained model
model = load_model('output/model.keras')

# Load the JSON file
with open("output/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Define the classes (update this list based on your model's classes)
class_names = list(class_indices.keys())  # Add your actual class names

# Create a simplified mapping for prediction display
simplified_classes = {}
for name in class_names:
    if 'apple' in name.lower() or 'Apple' in name:
        simplified_classes[name] = 'apple'
    elif 'tomato' in name.lower() or 'Tomato' in name:
        simplified_classes[name] = 'tomato'
    else:
        simplified_classes[name] = name  # Keep original if not apple or tomato

print("Original classes:", class_names)
print("Simplified to:", set(simplified_classes.values()))

# Create a mapping that preserves the original indices
indexed_simple_classes = {}
for name, idx in class_indices.items():
    indexed_simple_classes[idx] = simplified_classes[name]

# Create the final class_names array that maintains the original positions
class_names = [indexed_simple_classes[i] for i in range(len(class_indices))]

print("Final class names for display (preserving indices):", class_names)

# Define the upload folder
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the image size
IMG_SIZE = 128  # Must match what was used during training

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
    return render_template('index.html', history=prediction_history[:5])

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
        img_array, _ = safe_load_image(file_path, target_size=(IMG_SIZE, IMG_SIZE))

        # Make prediction
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_idx]
        confidence = float(np.max(predictions[0]))
        
        # Apply confidence threshold
        confidence_message = f"{confidence:.1%}"
        
        # Get original class name for debugging
        original_class_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_idx)]
        print(f"Prediction: {original_class_name} ({predicted_class}) with confidence: {confidence:.4f}")
        
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
            'original_class': original_class_name
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
                            original_class=original_class_name,
                            image_path=image_url,
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
    img_array, _ = safe_load_image(file_path, target_size=(IMG_SIZE, IMG_SIZE))

    # Make prediction with more detail
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = float(np.max(predictions[0]))
    
    # Get detailed class information
    original_class_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_idx)]
    
    # Log all tomato class probabilities for debugging
    print("All class probabilities:")
    for i, prob in enumerate(predictions[0]):
        if prob > 0.001:  # Only show non-zero probabilities
            class_name = list(class_indices.keys())[list(class_indices.values()).index(i)]
            simplified = simplified_classes[class_name]
            print(f"  {class_name} ({simplified}): {prob:.4f}")
    
    # Apply confidence threshold for better reliability
    # If confidence is too low, provide feedback about uncertainty
    if confidence < 0.5:
        prediction_message = f"Uncertain, but looks like a {predicted_class}"
    else:
        prediction_message = predicted_class
    
    # Log prediction details for debugging
    print(f"Prediction: {original_class_name} ({predicted_class}) with confidence: {confidence:.4f}")
    
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
        'original_class': original_class_name
    }
    
    # Add to history
    prediction_history.insert(0, history_entry)
    if len(prediction_history) > 20:
        prediction_history.pop()
    
    # Return detailed response
    return jsonify({
        'prediction': prediction_message,
        'confidence': confidence,
        'confidence_percent': f"{confidence:.1%}",
        'class_index': int(predicted_idx),
        'original_class': original_class_name,
        'all_confidences': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names)) 
                           if predictions[0][i] > 0.01}  # Only return significant confidences
    })

# New routes for enhanced features

@app.route('/history')
def view_history():
    return render_template('history.html', history=prediction_history)

@app.route('/analysis/<int:prediction_id>')
def analysis(prediction_id):
    # Find the prediction in history
    prediction_entry = None
    for entry in prediction_history:
        if entry['id'] == prediction_id:
            prediction_entry = entry
            break
    
    if not prediction_entry:
        return redirect(url_for('home'))
    
    file_path = prediction_entry['file_path']
    
    try:
        # Load and preprocess the image using safe_load_image
        img_array, _ = safe_load_image(file_path, target_size=(IMG_SIZE, IMG_SIZE))

        # Get all class probabilities
        predictions = model.predict(img_array)
        
        # Get all predictions sorted by confidence
        all_predictions = []
        for i, prob in enumerate(predictions[0]):
            if prob > 0.001:  # Only include non-zero probabilities
                class_name = list(class_indices.keys())[list(class_indices.values()).index(i)]
                simplified = simplified_classes[class_name]
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
        
        return render_template('analysis.html', 
                              prediction=prediction_entry,
                              top_predictions=top_predictions,
                              all_predictions=all_predictions,
                              image_path=image_url)
    except Exception as e:
        print(f"Error in analysis: {e}")
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

if __name__ == '__main__':
    app.run(debug=True)