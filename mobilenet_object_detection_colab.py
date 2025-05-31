import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import os
import json
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import gc
from PIL import Image

class MobileNetObjectDetectorColab:
    def __init__(self, img_size=224, num_classes=None, batch_size=8):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_indices = {}
        self.class_names = []
        self.batch_size = batch_size  # Smaller batch size for Colab
        
        # Create output directory
        self.output_dir = "output_mobilenet_detection"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure TensorFlow for memory efficiency
        self.configure_tf_memory()
        
    def configure_tf_memory(self):
        """Configure TensorFlow for memory efficiency"""
        try:
            # Enable memory growth for GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for memory efficiency
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
        except Exception as e:
            print(f"GPU configuration warning: {e}")
    
    def create_synthetic_bboxes(self, img_shape, class_name):
        """
        Create synthetic bounding boxes without loading image
        More memory efficient approach
        """
        h, w = img_shape[:2]
        
        # Create varied synthetic bounding boxes
        margin_x = np.random.uniform(0.05, 0.2)
        margin_y = np.random.uniform(0.05, 0.2)
        
        x_min = int(w * margin_x)
        y_min = int(h * margin_y)
        x_max = int(w * (1 - margin_x))
        y_max = int(h * (1 - margin_y))
        
        # Add noise
        noise_x = int(w * 0.02)
        noise_y = int(h * 0.02)
        
        x_min = max(0, x_min + np.random.randint(-noise_x, noise_x))
        y_min = max(0, y_min + np.random.randint(-noise_y, noise_y))
        x_max = min(w, x_max + np.random.randint(-noise_x, noise_x))
        y_max = min(h, y_max + np.random.randint(-noise_y, noise_y))
        
        # Normalize coordinates
        bbox = [x_min / w, y_min / h, x_max / w, y_max / h]
        class_id = self.class_indices.get(class_name, 0)
        
        return bbox, class_id
    
    def create_detection_model(self):
        """Create a memory-efficient MobileNetV2-based model"""
        # Input layer with explicit shape
        input_tensor = Input(shape=(self.img_size, self.img_size, 3), name='input_image')
        
        # Base model - MobileNetV2 with explicit input_shape parameter
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor,
            input_shape=(self.img_size, self.img_size, 3),  # Explicitly specify input shape
            alpha=0.75  # Reduced width multiplier for memory efficiency
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Smaller detection head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', name='feature_dense_1')(x)  # Reduced from 512
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='feature_dense_2')(x)  # Reduced from 256
        x = Dropout(0.3)(x)
        
        # Classification branch
        classification = Dense(
            self.num_classes, 
            activation='softmax', 
            name='classification',
            dtype='float32'  # Ensure float32 output
        )(x)
        
        # Bounding box regression branch
        bbox_regression = Dense(
            4, 
            activation='sigmoid', 
            name='bbox_regression',
            dtype='float32'  # Ensure float32 output
        )(x)
        
        # Create model
        self.model = Model(inputs=input_tensor, 
                          outputs=[classification, bbox_regression])
        
        # Compile with mixed precision compatible optimizer
        optimizer = Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'bbox_regression': 'mse'
            },
            loss_weights={
                'classification': 1.0,
                'bbox_regression': 3.0  # Reduced weight
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
        train_dir = os.path.join(data_dir, 'Training')
        if os.path.exists(train_dir):
            classes = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        
        self.class_indices = {class_name: idx for idx, class_name in enumerate(sorted(classes))}
        self.class_names = sorted(classes)
        self.num_classes = len(classes)
        
        print(f"Found {self.num_classes} classes: {self.class_names}")
        return self.class_indices
    
    def data_generator(self, data_dir, split='Training', max_samples_per_class=None):
        """Memory-efficient data generator"""
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            return
        
        image_paths = []
        class_names = []
        
        # Collect image paths
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit samples per class to save memory
            if max_samples_per_class:
                images = images[:max_samples_per_class]
            
            for img_file in images:
                image_paths.append(os.path.join(class_dir, img_file))
                class_names.append(class_name)
        
        # Shuffle data
        combined = list(zip(image_paths, class_names))
        random.shuffle(combined)
        image_paths, class_names = zip(*combined)
        
        # Generate batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_classes = class_names[i:i + self.batch_size]
            
            batch_images = []
            batch_bboxes = []
            batch_class_ids = []
            
            for img_path, class_name in zip(batch_paths, batch_classes):
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    original_shape = img.shape
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = img.astype(np.float32) / 255.0
                    
                    # Create synthetic bbox
                    bbox, class_id = self.create_synthetic_bboxes(original_shape, class_name)
                    
                    batch_images.append(img)
                    batch_bboxes.append(bbox)
                    batch_class_ids.append(class_id)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if batch_images:
                yield (
                    np.array(batch_images, dtype=np.float32),
                    {
                        'classification': np.array(batch_class_ids, dtype=np.int32),
                        'bbox_regression': np.array(batch_bboxes, dtype=np.float32)
                    }
                )
            
            # Clear memory
            del batch_images, batch_bboxes, batch_class_ids
            gc.collect()
    
    def prepare_dataset(self, data_dir, max_samples_per_class=200):
        """Prepare dataset with memory constraints"""
        self.get_class_indices_from_directory(data_dir)
        
        # Create smaller datasets for Colab
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(data_dir, 'Training', max_samples_per_class),
            output_signature=(
                tf.TensorSpec(shape=(None, self.img_size, self.img_size, 3), dtype=tf.float32),
                {
                    'classification': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    'bbox_regression': tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
                }
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(data_dir, 'Validation', max_samples_per_class//2),
            output_signature=(
                tf.TensorSpec(shape=(None, self.img_size, self.img_size, 3), dtype=tf.float32),
                {
                    'classification': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    'bbox_regression': tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
                }
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def train_detector(self, data_dir, epochs=20, max_samples_per_class=150):
        """Train with memory-efficient approach"""
        print("Preparing detection data...")
        
        train_dataset, val_dataset = self.prepare_dataset(data_dir, max_samples_per_class)
        
        # Create model
        self.create_detection_model()
        
        # Calculate steps per epoch (estimate)
        steps_per_epoch = (max_samples_per_class * self.num_classes) // self.batch_size
        validation_steps = (max_samples_per_class * self.num_classes // 2) // self.batch_size
        
        print(f"Estimated steps per epoch: {steps_per_epoch}")
        print(f"Estimated validation steps: {validation_steps}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,  # Reduced patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Reduced patience
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("Training object detection model...")
        
        # Train model
        history = self.model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=epochs,
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
        """Plot training history with memory cleanup"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Smaller figure
        
        # Classification accuracy
        axes[0, 0].plot(history.history['classification_accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(history.history['val_classification_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Classification Accuracy', fontsize=10)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification loss
        axes[0, 1].plot(history.history['classification_loss'], label='Training', linewidth=2)
        axes[0, 1].plot(history.history['val_classification_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Classification Loss', fontsize=10)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bounding box MAE
        axes[1, 0].plot(history.history['bbox_regression_mae'], label='Training', linewidth=2)
        axes[1, 0].plot(history.history['val_bbox_regression_mae'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Bounding Box MAE', fontsize=10)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total loss
        axes[1, 1].plot(history.history['loss'], label='Training', linewidth=2)
        axes[1, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Total Loss', fontsize=10)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_history.png', dpi=150, bbox_inches='tight')  # Lower DPI
        plt.show()
        plt.close()
        
        # Clear memory
        del fig, axes
        gc.collect()
    
    def predict_with_bbox(self, image_path):
        """Memory-efficient prediction"""
        try:
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
            
            # Clear memory
            del img_batch, class_pred, bbox_pred
            gc.collect()
            
            return original_img, class_name, confidence, (x_min, y_min, x_max, y_max)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, None, None
    
    def evaluate_detection_memory_efficient(self, data_dir, max_samples=100):
        """Memory-efficient evaluation"""
        print("Evaluating detection model...")
        
        test_dir = os.path.join(data_dir, 'Test')
        if not os.path.exists(test_dir):
            print("Test directory not found")
            return None, None, {}
        
        all_predictions = []
        all_true_classes = []
        all_ious = []
        
        sample_count = 0
        
        for class_name in self.class_names:
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit samples for memory
            images = images[:max_samples // len(self.class_names)]
            
            for img_file in images:
                if sample_count >= max_samples:
                    break
                    
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # Load and predict
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    original_shape = img.shape
                    img_resized = cv2.resize(img, (self.img_size, self.img_size))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    img_batch = np.expand_dims(img_normalized, axis=0)
                    
                    # Predict
                    class_pred, bbox_pred = self.model.predict(img_batch, verbose=0)
                    predicted_class = np.argmax(class_pred[0])
                    
                    # True values
                    true_class = self.class_indices[class_name]
                    true_bbox, _ = self.create_synthetic_bboxes(original_shape, class_name)
                    
                    # Calculate IoU
                    iou = self.calculate_iou(bbox_pred[0], true_bbox)
                    
                    all_predictions.append(predicted_class)
                    all_true_classes.append(true_class)
                    all_ious.append(iou)
                    
                    sample_count += 1
                    
                    # Clear memory
                    del img, img_batch, class_pred, bbox_pred
                    
                except Exception as e:
                    print(f"Error evaluating {img_path}: {e}")
                    continue
            
            if sample_count >= max_samples:
                break
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_true_classes))
        mean_iou = np.mean(all_ious)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_true_classes, all_predictions)
        
        # Save simplified metrics
        metrics = {
            "classification_accuracy": float(accuracy),
            "mean_iou": float(mean_iou),
            "num_classes": self.num_classes,
            "test_samples": len(all_predictions)
        }
        
        with open(f'{self.output_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot smaller confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Classification Accuracy: {accuracy:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        
        # Clear memory
        gc.collect()
        
        return accuracy, mean_iou, metrics
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

def visualize_detection_results_colab(detector, test_dir, num_samples=4):
    """Memory-efficient visualization for Colab"""
    print("Creating detection visualizations...")
    
    sample_images = []
    sample_classes = []
    
    # Get fewer samples
    for class_name in detector.class_names[:2]:  # Only 2 classes
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                for img in images[:2]:  # Only 2 per class
                    sample_images.append(os.path.join(class_dir, img))
                    sample_classes.append(class_name)
    
    sample_images = sample_images[:num_samples]
    sample_classes = sample_classes[:num_samples]
    
    if not sample_images:
        print("No sample images found")
        return
    
    # Create smaller visualization
    fig, axes = plt.subplots(1, len(sample_images), figsize=(12, 3))
    if len(sample_images) == 1:
        axes = [axes]
    
    for i, (img_path, true_class) in enumerate(zip(sample_images, sample_classes)):
        result_img, pred_class, confidence, bbox = detector.predict_with_bbox(img_path)
        
        if result_img is not None:
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(result_img_rgb)
            axes[i].set_title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{detector.output_dir}/detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Clear memory
    del fig, axes
    gc.collect()

# Colab-optimized training script
if __name__ == "__main__":
    print("Starting MobileNetV2 Object Detection Training (Colab Optimized)...")
    
    # Create detector with Colab-friendly settings
    detector = MobileNetObjectDetectorColab(img_size=224, batch_size=8)
    
    # Train with reduced parameters
    print("Training MobileNetV2 object detection model...")
    history = detector.train_detector("working", epochs=15, max_samples_per_class=100)
    
    # Plot training history
    detector.plot_training_history(history)
    
    # Evaluate model with memory constraints
    print("Evaluating model...")
    accuracy, mean_iou, metrics = detector.evaluate_detection_memory_efficient("working", max_samples=50)
    
    # Visualize results
    test_dir = "working/Test"
    if os.path.exists(test_dir):
        visualize_detection_results_colab(detector, test_dir, num_samples=4)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {detector.output_dir}/")
    if accuracy is not None:
        print(f"Final Classification Accuracy: {accuracy:.4f}")
        print(f"Final Mean IoU: {mean_iou:.4f}")
    print(f"Classes detected: {len(detector.class_names)}")
    
    # Final memory cleanup
    gc.collect()