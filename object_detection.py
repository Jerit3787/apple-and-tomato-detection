import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import random

class ObjectDetector:
    def __init__(self, img_size=416, num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes  # apple, tomato
        self.model = None
        
        # Create output directory
        os.makedirs("output_detection", exist_ok=True)
        
    def create_synthetic_bboxes(self, image_path, class_name):
        """
        Create synthetic bounding boxes for training
        In a real scenario, you'd have actual annotations
        """
        # Load image
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Create a synthetic bounding box covering most of the fruit
        # This is a simplified approach - in practice you'd have real annotations
        margin = 0.1
        x_min = int(w * margin)
        y_min = int(h * margin)
        x_max = int(w * (1 - margin))
        y_max = int(h * (1 - margin))
        
        # Normalize coordinates
        bbox = [x_min / w, y_min / h, x_max / w, y_max / h]
        
        # Class encoding: 0 for apple, 1 for tomato
        class_id = 0 if 'apple' in class_name.lower() else 1
        
        return bbox, class_id
    
    def create_detection_model(self):
        """Create a model for object detection with classification and localization"""
        # Base model
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add detection head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Classification branch
        classification = Dense(self.num_classes, activation='softmax', name='classification')(x)
        
        # Bounding box regression branch (4 coordinates: x_min, y_min, x_max, y_max)
        bbox_regression = Dense(4, activation='sigmoid', name='bbox_regression')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, 
                          outputs=[classification, bbox_regression])
        
        # Compile with multiple losses
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'bbox_regression': 'mse'
            },
            loss_weights={
                'classification': 1.0,
                'bbox_regression': 10.0  # Higher weight for localization
            },
            metrics={
                'classification': ['accuracy'],
                'bbox_regression': ['mae']
            }
        )
        
        return self.model
    
    def prepare_detection_data(self, data_dir):
        """Prepare data for object detection training"""
        images = []
        bboxes = []
        class_ids = []
        
        # Walk through directory structure
        for split in ['Training', 'Validation', 'Test']:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img.astype(np.float32) / 255.0
                        
                        # Create synthetic bbox and class
                        bbox, class_id = self.create_synthetic_bboxes(img_path, class_name)
                        
                        images.append(img)
                        bboxes.append(bbox)
                        class_ids.append(class_id)
        
        return np.array(images), np.array(bboxes), np.array(class_ids)
    
    def train_detector(self, data_dir, epochs=30):
        """Train the object detection model"""
        print("Preparing detection data...")
        images, bboxes, class_ids = self.prepare_detection_data(data_dir)
        
        print(f"Total samples: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"BBox shape: {bboxes[0].shape}")
        
        # Split data
        split_idx = int(0.8 * len(images))
        
        train_images = images[:split_idx]
        train_bboxes = bboxes[:split_idx]
        train_classes = class_ids[:split_idx]
        
        val_images = images[split_idx:]
        val_bboxes = bboxes[split_idx:]
        val_classes = class_ids[split_idx:]
        
        # Create model
        self.create_detection_model()
        
        print("Training object detection model...")
        
        # Train model
        history = self.model.fit(
            train_images,
            {
                'classification': train_classes,
                'bbox_regression': train_bboxes
            },
            validation_data=(
                val_images,
                {
                    'classification': val_classes,
                    'bbox_regression': val_bboxes
                }
            ),
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        
        # Save model
        self.model.save('output_detection/detection_model.keras')
        
        # Save training history
        with open('output_detection/detection_history.json', 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
        
        return history
    
    def predict_with_bbox(self, image_path):
        """Make prediction with bounding box visualization"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        original_img = img.copy()
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        class_pred, bbox_pred = self.model.predict(img_batch)
        
        # Get predicted class
        predicted_class = np.argmax(class_pred[0])
        confidence = class_pred[0][predicted_class]
        class_name = 'Apple' if predicted_class == 0 else 'Tomato'
        
        # Get predicted bounding box
        bbox = bbox_pred[0]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = original_img.shape[:2]
        x_min = int(bbox[0] * w)
        y_min = int(bbox[1] * h)
        x_max = int(bbox[2] * w)
        y_max = int(bbox[3] * h)
        
        # Draw bounding box and label
        cv2.rectangle(original_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(original_img, label, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return original_img, class_name, confidence, (x_min, y_min, x_max, y_max)
    
    def evaluate_detection(self, test_images, test_bboxes, test_classes):
        """Evaluate detection model"""
        predictions = self.model.predict(test_images)
        class_preds, bbox_preds = predictions
        
        # Classification accuracy
        predicted_classes = np.argmax(class_preds, axis=1)
        accuracy = np.mean(predicted_classes == test_classes)
        
        # IoU calculation for bounding boxes
        ious = []
        for i in range(len(test_bboxes)):
            pred_bbox = bbox_preds[i]
            true_bbox = test_bboxes[i]
            iou = self.calculate_iou(pred_bbox, true_bbox)
            ious.append(iou)
        
        mean_iou = np.mean(ious)
        
        print(f"Detection Accuracy: {accuracy:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        
        return accuracy, mean_iou
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

def visualize_detection_results(detector, test_dir, num_samples=5):
    """Visualize detection results on sample images"""
    # Get sample images
    sample_images = []
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images[:2]:  # Take 2 from each class
                sample_images.append(os.path.join(class_dir, img))
    
    # Limit to num_samples
    sample_images = sample_images[:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(1, len(sample_images), figsize=(20, 4))
    if len(sample_images) == 1:
        axes = [axes]
    
    for i, img_path in enumerate(sample_images):
        # Get detection result
        result_img, class_name, confidence, bbox = detector.predict_with_bbox(img_path)
        
        # Convert BGR to RGB for matplotlib
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Display
        axes[i].imshow(result_img_rgb)
        axes[i].set_title(f"Predicted: {class_name}\nConfidence: {confidence:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('output_detection/detection_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create object detector
    detector = ObjectDetector(img_size=224, num_classes=2)
    
    # Train detector
    print("Training object detection model...")
    history = detector.train_detector("working", epochs=20)
    
    # Visualize results
    print("Creating visualization...")
    visualize_detection_results(detector, "working/Test", num_samples=6)
    
    print("Object detection training completed!")
    print("Check output_detection/ directory for results.")