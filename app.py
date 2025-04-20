import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('output/model.keras')

# Define the classes (update this list based on your model's classes)
class_names = ['Nerodia sipedon - Northern Watersnake', 'Thamnophis sirtalis - Common Garter snake', "Storeria dekayi - DeKay's Brown snake", "Patherophis obsoletus - Black Rat snake", "Crotalus atrox - Western Diamondback rattlesnake"]  # Add your actual class names

# Define the upload folder
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    img = image.load_img(file_path, target_size=(224, 224))  # Adjust size based on your model
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    return render_template('index.html', prediction=predicted_class, number=np.argmax(predictions[0]))

if __name__ == '__main__':
    app.run(debug=True)