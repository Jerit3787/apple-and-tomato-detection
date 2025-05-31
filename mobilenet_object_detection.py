import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import random
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

class MobileNetObjectDetector:
    def __init__(self, img_size=224, num_classes=None):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_indices = {}
        self.class_names = []
        
        # Create output directory
        self.output_dir = "output_mobilenet_detection"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_synthetic_bboxes(self, image_path, class_name):
        """
        Create synthetic bounding boxes for training
        Uses a more realistic approach with some variation
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        h, w = img.shape[:2]
        
        # Create more varied synthetic bounding boxes
        # Add some randomness to make training more robust
        margin_x = np.random.uniform(0.05, 0.15)
        margin_y = np.random.uniform(0.05, 0.15)
        
        x_min = int(w * margin_x)
        y_min = int(h * margin_y)
        x_max = int(w * (1 - margin_x))
        y_max = int(h * (1 - margin_y))
        
        # Add some noise to make it more realistic
        noise_x = int(w * 0.02)
        noise_y = int(h * 0.02)
        
        x_min = max(0, x_min + np.random.randint(-noise_x, noise_x))
        y_min = max(0, y_min + np.random.randint(-noise_y, noise_y))
        x_max = min(w, x_max + np.random.randint(-noise_x, noise_x))
        y_max = min(h, y_max + np.random.randint(-noise_y, noise_y))
        
        # Normalize coordinates
        bbox = [x_min / w, y_min / h, x_max / w, y_max / h]
        
        # Get class ID from class_indices
        class_id = self.class_indices.get(class_name, 0)
        
        return bbox, class_id
    
    def create_detection_model(self):
        """Create a MobileNetV2-based model for object detection"""
        # Input layer
        input_tensor = Input(shape=(self.img_size, self.img_size, 3))
        
        # Base model - MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor,
            alpha=1.0  # Width multiplier
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add detection head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='feature_dense_1')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='feature_dense_2')(x)
        x = Dropout(0.3)(x)
        
        # Classification branch
        classification = Dense(
            self.num_classes, 
            activation='softmax', 
            name='classification'
        )(x)
        
        # Bounding box regression branch (4 coordinates: x_min, y_min, x_max, y_max)
        bbox_regression = Dense(
            4, 
            activation='sigmoid', 
            name='bbox_regression'
        )(x)
        
        # Create model
        self.model = Model(inputs=input_tensor, 
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
                'bbox_regression': 5.0  # Weight for localization
            },
            metrics={
                'classification': ['accuracy'],
                'bbox_regression': ['mae']
            }
        )
        
        return self.model
    
    def get_class_indices_from_directory(self, data_dir):
        """Get class indices from directory structure"""
        classes = []
        
        # Check Training directory for classes
        train_dir = os.path.join(data_dir, 'Training')
        if os.path.exists(train_dir):
            classes = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        
        # Create class indices dictionary
        self.class_indices = {class_name: idx for idx, class_name in enumerate(sorted(classes))}
        self.class_names = sorted(classes)
        self.num_classes = len(classes)
        
        print(f"Found {self.num_classes} classes: {self.class_names}")
        return self.class_indices
    
    def prepare_detection_data(self, data_dir):
        """Prepare data for object detection training"""
        images = []
        bboxes = []
        class_ids = []
        
        # Get class indices first
        self.get_class_indices_from_directory(data_dir)
        
        # Walk through directory structure
        for split in ['Training', 'Validation', 'Test']:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            print(f"Processing {split} data...")
            split_count = 0
            
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                class_count = 0
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
                        if bbox is None:
                            continue
                        
                        images.append(img)
                        bboxes.append(bbox)
                        class_ids.append(class_id)
                        class_count += 1
                        split_count += 1
                
                print(f"  {class_name}: {class_count} images")
            
            print(f"Total {split} samples: {split_count}")
        
        print(f"Total dataset size: {len(images)} images")
        return np.array(images), np.array(bboxes), np.array(class_ids)
    
    def train_detector(self, data_dir, epochs=50):
        """Train the object detection model"""
        print("Preparing detection data...")
        images, bboxes, class_ids = self.prepare_detection_data(data_dir)
        
        print(f"Total samples: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"BBox shape: {bboxes[0].shape}")
        print(f"Classes: {self.num_classes}")
        
        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(images))
        
        train_images = images[:split_idx]
        train_bboxes = bboxes[:split_idx]
        train_classes = class_ids[:split_idx]
        
        val_images = images[split_idx:]
        val_bboxes = bboxes[split_idx:]
        val_classes = class_ids[split_idx:]
        
        print(f"Training samples: {len(train_images)}")
        print(f"Validation samples: {len(val_images)}")
        
        # Create model
        self.create_detection_model()
        
        # Print model summary
        print("\nModel Summary:")
        self.model.summary()
        
        # Save model architecture
        with open(f'{self.output_dir}/model_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
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
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        self.model.save(f'{self.output_dir}/mobilenet_detection_model.keras')
        
        # Save training history
        with open(f'{self.output_dir}/detection_history.json', 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
        
        # Save class indices
        with open(f'{self.output_dir}/class_indices.json', 'w') as f:
            json.dump(self.class_indices, f, indent=2)
        
        return history
    
    def plot_training_history(self, history):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Classification accuracy
        axes[0, 0].plot(history.history['classification_accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_classification_accuracy'], label='Validation')
        axes[0, 0].set_title('Classification Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification loss
        axes[0, 1].plot(history.history['classification_loss'], label='Training')
        axes[0, 1].plot(history.history['val_classification_loss'], label='Validation')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bounding box MAE
        axes[1, 0].plot(history.history['bbox_regression_mae'], label='Training')
        axes[1, 0].plot(history.history['val_bbox_regression_mae'], label='Validation')
        axes[1, 0].set_title('Bounding Box MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total loss
        axes[1, 1].plot(history.history['loss'], label='Training')
        axes[1, 1].plot(history.history['val_loss'], label='Validation')
        axes[1, 1].set_title('Total Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def predict_with_bbox(self, image_path):
        """Make prediction with bounding box visualization"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, None
            
        original_img = img.copy()
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        class_pred, bbox_pred = self.model.predict(img_batch, verbose=0)
        
        # Get predicted class
        predicted_class = np.argmax(class_pred[0])
        confidence = class_pred[0][predicted_class]
        class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else "Unknown"
        
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
        print("Evaluating detection model...")
        
        predictions = self.model.predict(test_images, verbose=1)
        class_preds, bbox_preds = predictions
        
        # Classification metrics
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
        
        # Generate confusion matrix
        cm = confusion_matrix(test_classes, predicted_classes)
        
        # Classification report
        report = classification_report(
            test_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save metrics
        metrics = {
            "classification_accuracy": float(accuracy),
            "mean_iou": float(mean_iou),
            "num_classes": self.num_classes,
            "test_samples": len(test_images),
            "classification_report": report
        }
        
        with open(f'{self.output_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save classification report
        report_text = classification_report(
            test_classes, 
            predicted_classes, 
            target_names=self.class_names
        )
        
        with open(f'{self.output_dir}/classification_report.txt', 'w') as f:
            f.write("MOBILENET OBJECT DETECTION EVALUATION\n")
            f.write("="*50 + "\n\n")
            f.write(f"Classification Accuracy: {accuracy:.4f}\n")
            f.write(f"Mean IoU: {mean_iou:.4f}\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
            f.write(f"Test Samples: {len(test_images)}\n\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write("-"*30 + "\n")
            f.write(report_text)
        
        print(f"Classification Accuracy: {accuracy:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        
        return accuracy, mean_iou, metrics
    
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

def visualize_detection_results(detector, test_dir, num_samples=8):
    """Visualize detection results on sample images"""
    print("Creating detection visualizations...")
    
    # Get sample images from different classes
    sample_images = []
    sample_classes = []
    
    for class_name in detector.class_names[:4]:  # Limit to first 4 classes for display
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                # Take 2 samples from each class
                for img in images[:2]:
                    sample_images.append(os.path.join(class_dir, img))
                    sample_classes.append(class_name)
    
    # Limit to num_samples
    sample_images = sample_images[:num_samples]
    sample_classes = sample_classes[:num_samples]
    
    if not sample_images:
        print("No sample images found for visualization")
        return
    
    # Create visualization
    rows = 2
    cols = min(4, len(sample_images))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (img_path, true_class) in enumerate(zip(sample_images, sample_classes)):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        # Get detection result
        result_img, pred_class, confidence, bbox = detector.predict_with_bbox(img_path)
        
        if result_img is not None:
            # Convert BGR to RGB for matplotlib
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # Display
            axes[row, col].imshow(result_img_rgb)
            axes[row, col].set_title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}")
            axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, 'Image\nLoad Error', 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_images), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{detector.output_dir}/detection_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {detector.output_dir}/detection_results.png")

if __name__ == "__main__":
    print("Starting MobileNetV2 Object Detection Training...")
    
    # Create object detector
    detector = MobileNetObjectDetector(img_size=224)
    
    # Train detector
    print("Training MobileNetV2 object detection model...")
    history = detector.train_detector("working", epochs=30)
    
    # Plot training history
    detector.plot_training_history(history)
    
    # Prepare test data for evaluation
    print("Preparing test data for evaluation...")
    test_images, test_bboxes, test_classes = detector.prepare_detection_data("working")
    
    # Use only test split for evaluation
    test_split_start = int(0.8 * len(test_images))
    eval_images = test_images[test_split_start:]
    eval_bboxes = test_bboxes[test_split_start:]
    eval_classes = test_classes[test_split_start:]
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, mean_iou, metrics = detector.evaluate_detection(eval_images, eval_bboxes, eval_classes)
    
    # Visualize results
    print("Creating detection visualizations...")
    test_dir = "working/Test"
    if os.path.exists(test_dir):
        visualize_detection_results(detector, test_dir, num_samples=8)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {detector.output_dir}/")
    print(f"Final Classification Accuracy: {accuracy:.4f}")
    print(f"Final Mean IoU: {mean_iou:.4f}")
    print(f"Classes detected: {len(detector.class_names)}")