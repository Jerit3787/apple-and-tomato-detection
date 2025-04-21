import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('output/model.keras')

import json

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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

    return render_template('index.html', 
                          prediction=predicted_class, 
                          confidence=confidence_message,
                          number=predicted_idx,
                          original_class=original_class_name)

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
    img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction with more detail
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = float(np.max(predictions[0]))
    
    # Get detailed class information
    original_class_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_idx)]
    
    # Apply confidence threshold for better reliability
    # If confidence is too low, provide feedback about uncertainty
    if confidence < 0.5:
        prediction_message = f"Uncertain, but looks like a {predicted_class}"
    else:
        prediction_message = predicted_class
    
    # Log prediction details for debugging
    print(f"Prediction: {original_class_name} ({predicted_class}) with confidence: {confidence:.4f}")
    
    # Return detailed response
    return jsonify({
        'prediction': prediction_message,
        'confidence': confidence,
        'class_index': int(predicted_idx),
        'original_class': original_class_name,
        'all_confidences': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names)) 
                           if predictions[0][i] > 0.01}  # Only return significant confidences
    })

if __name__ == '__main__':
    app.run(debug=True)