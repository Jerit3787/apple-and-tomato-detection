import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

def extract_detection_model_classes():
    """
    Extract class indices from a trained detection model
    without retraining the model. This script specifically
    handles the detection model in the output_detection folder.
    """
    print("Extracting class indices from detection model...")
    
    # Load the existing detection model
    model_path = 'output_detection/detection_model.keras'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # Load the model
        detection_model = load_model(model_path)
        print("Successfully loaded detection model")
        
        # Create class indices mapping
        # For the detection model, the mapping is hardcoded as:
        # 0: apple, 1: tomato (as per object_detection.py)
        class_indices = {
            "Apple": 0,
            "Tomato": 1
        }
        
        # Create simplified class mapping
        simplified_classes = {
            "Apple": "apple",
            "Tomato": "tomato"
        }
        
        # Output directory should exist, but check just in case
        os.makedirs("output_detection", exist_ok=True)
        
        # Save class indices to JSON file
        with open("output_detection/class_indices.json", "w") as f:
            json.dump(class_indices, f, indent=2)
        print("Saved class indices to output_detection/class_indices.json")
        
        # Also save the simplified classes
        with open("output_detection/simplified_classes.json", "w") as f:
            json.dump(simplified_classes, f, indent=2)
        print("Saved simplified classes to output_detection/simplified_classes.json")
        
        # Verify the model outputs
        print("\nVerifying model structure:")
        print(f"Model inputs: {detection_model.inputs}")
        print(f"Model outputs: {detection_model.outputs}")
        print(f"Output shapes: {[output.shape for output in detection_model.outputs]}")
        
        return True
    
    except Exception as e:
        print(f"Error loading or processing the model: {str(e)}")
        return False

def test_detection_model():
    """Test the detection model with a sample image to verify class indices"""
    # Find a sample image in the uploads folder
    upload_dir = 'uploads'
    sample_image = None
    
    for filename in os.listdir(upload_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_image = os.path.join(upload_dir, filename)
            break
    
    if not sample_image:
        print("No sample images found in uploads folder")
        return
    
    try:
        # Load the model
        model = load_model('output_detection/detection_model.keras')
        
        # Load and preprocess the image
        img = cv2.imread(sample_image)
        img_resized = cv2.resize(img, (224, 224))  # Match model's expected size
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch)
        
        # Parse predictions
        class_pred = predictions[0] if isinstance(predictions, list) else predictions
        predicted_class_idx = np.argmax(class_pred[0])
        confidence = class_pred[0][predicted_class_idx]
        
        # Load class indices
        with open('output_detection/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        # Get class name
        class_name = None
        for name, idx in class_indices.items():
            if idx == predicted_class_idx:
                class_name = name
                break
                
        print(f"\nDetection test results for {os.path.basename(sample_image)}:")
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Predicted class: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error testing detection model: {str(e)}")

if __name__ == "__main__":
    # Extract class indices from the detection model
    if extract_detection_model_classes():
        print("\nSuccessfully extracted class information from detection model")
        
        # Test the model with a sample image
        test_detection_model()
    else:
        print("\nFailed to extract class information from detection model")