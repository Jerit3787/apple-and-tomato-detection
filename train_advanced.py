import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, F1Score
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AdvancedFruitClassifier:
    def __init__(self, img_size=224, batch_size=32, architecture='resnet50'):
        self.img_size = img_size
        self.batch_size = batch_size
        self.architecture = architecture
        self.model = None
        self.history = None
        
        # Directories
        self.train_dir = "working/Training"
        self.val_dir = "working/Validation"
        self.test_dir = "working/Test"
        
        # Create output directory
        os.makedirs("output_advanced", exist_ok=True)
        
    def create_data_generators(self):
        """Create enhanced data generators with advanced augmentation"""
        # Advanced augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1
        )
        
        # Simple rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Store class information
        self.class_indices = self.train_generator.class_indices
        self.num_classes = len(self.class_indices)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Class indices: {self.class_indices}")
        
    def create_model(self):
        """Create model with different CNN architectures"""
        if self.architecture == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif self.architecture == 'mobilenetv2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif self.architecture == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        else:
            raise ValueError("Unsupported architecture. Choose from: resnet50, mobilenetv2, vgg16")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), F1Score()]
        )
        
        print(f"Model created with {self.architecture} architecture")
        print(f"Total parameters: {self.model.count_params():,}")
        
    def train_model(self, epochs=50, fine_tune_epochs=20):
        """Train model with transfer learning and fine-tuning"""
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
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'output_advanced/best_model_{self.architecture}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calculate steps
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.val_generator.samples // self.batch_size
        
        print("Phase 1: Training with frozen base model...")
        
        # Initial training with frozen base
        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        if fine_tune_epochs > 0:
            print("\nPhase 2: Fine-tuning with unfrozen layers...")
            
            # Unfreeze some layers for fine-tuning
            if self.architecture == 'resnet50':
                # Unfreeze last 20 layers
                for layer in self.model.layers[-20:]:
                    layer.trainable = True
            elif self.architecture == 'mobilenetv2':
                # Unfreeze last 30 layers
                for layer in self.model.layers[-30:]:
                    layer.trainable = True
            elif self.architecture == 'vgg16':
                # Unfreeze last 10 layers
                for layer in self.model.layers[-10:]:
                    layer.trainable = True
            
            # Recompile with lower learning rate and consistent metric names
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']  # Simplified metrics to avoid naming conflicts
            )
            
            # Continue training
            history2 = self.model.fit(
                self.train_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.val_generator,
                validation_steps=validation_steps,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories safely
            combined_history = {}
            
            # Get common keys between both histories
            common_keys = set(history1.history.keys()) & set(history2.history.keys())
            
            for key in common_keys:
                combined_history[key] = history1.history[key] + history2.history[key]
            
            # Add keys that only exist in history1
            for key in history1.history.keys():
                if key not in combined_history:
                    combined_history[key] = history1.history[key] + [None] * len(history2.history['loss'])
            
            self.history = combined_history
        else:
            self.history = history1.history
            
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("Evaluating model on test set...")
        
        # Reset test generator
        self.test_generator.reset()
        
        # Get predictions
        test_steps = self.test_generator.samples // self.batch_size
        predictions = self.model.predict(self.test_generator, steps=test_steps, verbose=1)
        
        # Get true labels
        y_true = self.test_generator.classes[:len(predictions)]
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.model.evaluate(
            self.test_generator, steps=test_steps, verbose=1
        )
        
        # Class names
        class_names = list(self.class_indices.keys())
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Print results
        print(f"\nTest Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        
        # Save detailed classification report
        with open(f'output_advanced/classification_report_{self.architecture}.txt', 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.architecture.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'output_advanced/confusion_matrix_{self.architecture}.png', dpi=300)
        plt.close()
        
        # Save metrics
        metrics = {
            'architecture': self.architecture,
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1_score': float(test_f1),
            'test_loss': float(test_loss),
            'classification_report': report
        }
        
        with open(f'output_advanced/metrics_{self.architecture}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
        
    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(self.history['accuracy'], label='Training')
        axes[0,0].plot(self.history['val_accuracy'], label='Validation')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Loss
        axes[0,1].plot(self.history['loss'], label='Training')
        axes[0,1].plot(self.history['val_loss'], label='Validation')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Precision
        axes[1,0].plot(self.history['precision'], label='Precision')
        axes[1,0].plot(self.history['recall'], label='Recall')
        axes[1,0].plot(self.history['f1_score'], label='F1-Score')
        axes[1,0].set_title('Model Metrics')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Score')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history:
            axes[1,1].plot(self.history['lr'])
            axes[1,1].set_title('Learning Rate')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Learning Rate')
            axes[1,1].set_yscale('log')
        else:
            axes[1,1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'output_advanced/training_history_{self.architecture}.png', dpi=300)
        plt.close()
        
    def save_model(self):
        """Save the trained model and related files"""
        # Save model
        self.model.save(f'output_advanced/model_{self.architecture}.keras')
        
        # Save class indices
        with open(f'output_advanced/class_indices_{self.architecture}.json', 'w') as f:
            json.dump(self.class_indices, f, indent=2)
            
        # Save training history
        with open(f'output_advanced/history_{self.architecture}.json', 'w') as f:
            json.dump(self.history, f, indent=2)
            
        # Save model summary
        with open(f'output_advanced/model_summary_{self.architecture}.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
        print(f"Model and related files saved with prefix: {self.architecture}")

def run_experiment(architecture='resnet50', img_size=224, batch_size=32, epochs=30, fine_tune_epochs=10):
    """Run complete training experiment"""
    print(f"Starting experiment with {architecture}")
    print(f"Image size: {img_size}, Batch size: {batch_size}")
    print("="*50)
    
    # Create classifier
    classifier = AdvancedFruitClassifier(
        img_size=img_size,
        batch_size=batch_size,
        architecture=architecture
    )
    
    # Prepare data
    classifier.create_data_generators()
    
    # Create model
    classifier.create_model()
    
    # Train model
    classifier.train_model(epochs=epochs, fine_tune_epochs=fine_tune_epochs)
    
    # Evaluate model
    metrics = classifier.evaluate_model()
    
    # Plot and save results
    classifier.plot_training_history()
    classifier.save_model()
    
    return classifier, metrics

if __name__ == "__main__":
    # Run experiments with different architectures
    architectures = ['resnet50', 'mobilenetv2', 'vgg16']
    all_results = {}
    
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {arch.upper()}")
        print(f"{'='*60}")
        
        try:
            classifier, metrics = run_experiment(
                architecture=arch,
                img_size=224,
                batch_size=32,
                epochs=20,
                fine_tune_epochs=10
            )
            all_results[arch] = metrics
            
        except Exception as e:
            print(f"Error training {arch}: {str(e)}")
            continue
    
    # Compare results
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for arch, metrics in all_results.items():
        comparison_data.append({
            'Architecture': arch.upper(),
            'Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Precision': f"{metrics['test_precision']:.4f}",
            'Recall': f"{metrics['test_recall']:.4f}",
            'F1-Score': f"{metrics['test_f1_score']:.4f}"
        })
    
    # Save comparison
    with open('output_advanced/model_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Print comparison table
    print(f"{'Architecture':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for data in comparison_data:
        print(f"{data['Architecture']:<12} {data['Accuracy']:<10} {data['Precision']:<10} {data['Recall']:<10} {data['F1-Score']:<10}")
    
    print("\nTraining completed! Check output_advanced/ directory for results.")