import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import json
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directories
train_dir = "working/Training"
val_dir = "working/Validation"
test_dir = "working/Test"

# Create a simpler ImageDataGenerator with minimal augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Use smaller image size and larger batch size
IMG_SIZE = 128  # Reduced from 224
BATCH_SIZE = 64  # Increased from 32

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Print class information
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# Create a simpler CNN model
print("Creating model...")
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define callbacks - just early stopping to save resources
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

print("Training the model...")

# Steps per epoch - limit to 100 steps max to save resources
steps_per_epoch = min(100, train_generator.samples // BATCH_SIZE)
validation_steps = min(50, val_generator.samples // BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=20,  # Reduced from 50
    callbacks=callbacks,
    verbose=1
)

print("Evaluating the model...")

# Test on a limited number of batches
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_steps = min(30, test_generator.samples // BATCH_SIZE)
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Create output directory
os.makedirs("output", exist_ok=True)

# Save model as one file
model.save('output/model.keras')

# Save the class indices
with open("output/class_indices.json", "w") as json_file:
    json.dump(train_generator.class_indices, json_file)

# Create simplified class mapping
class_names = list(class_indices.keys())
simplified_classes = {}
for name in class_names:
    if 'apple' in name.lower() or 'Apple' in name:
        simplified_classes[name] = 'apple'
    elif 'tomato' in name.lower() or 'Tomato' in name:
        simplified_classes[name] = 'tomato'
    else:
        simplified_classes[name] = name

with open("output/simplified_classes.json", "w") as json_file:
    json.dump(simplified_classes, json_file)

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('output/training_history.png')
plt.close()

# Get predictions for confusion matrix and classification report
y_pred = []
y_true = []

# Reset generator to start from the beginning
test_generator.reset()
for i in range(test_steps):
    try:
        X_batch, y_batch = next(test_generator)
        batch_pred = model.predict(X_batch, verbose=0)
        # Convert predictions to class indices
        batch_pred_classes = np.argmax(batch_pred, axis=1)
        batch_true_classes = np.argmax(y_batch, axis=1)
        
        y_pred.extend(batch_pred_classes)
        y_true.extend(batch_true_classes)
    except StopIteration:
        break

# Generate classification report - fix the error with mismatched classes
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Get unique classes that actually appear in the test data
unique_classes = sorted(set(y_true))
# Use only the class labels that correspond to classes found in the test data
filtered_class_labels = [class_labels[i] for i in unique_classes]

# Create classification report with the correct target names
report = classification_report(y_true, y_pred, 
                               labels=unique_classes,  # Use only classes present in test data
                               target_names=filtered_class_labels)
print("\nClassification Report:")
print(report)

# Save report to file
with open('output/classification_report.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    f.write(report)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Add labels - use only classes present in test data
tick_marks = np.arange(len(filtered_class_labels))
plt.xticks(tick_marks, filtered_class_labels, rotation=90)
plt.yticks(tick_marks, filtered_class_labels)

# Add number labels to each cell
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('output/confusion_matrix.png')
plt.close()

# Create simplified confusion matrix (for Apple vs Tomato only)
simplified_y_true = ['apple' if simplified_classes[class_labels[i]] == 'apple' else 'tomato' for i in y_true]
simplified_y_pred = ['apple' if simplified_classes[class_labels[i]] == 'apple' else 'tomato' for i in y_pred]

# Create confusion matrix for apple vs tomato
from sklearn.metrics import confusion_matrix
simplified_cm = confusion_matrix(simplified_y_true, simplified_y_pred, labels=['apple', 'tomato'])

# Plot simplified confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(simplified_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Apple vs Tomato Confusion Matrix')
plt.colorbar()

# Add labels
plt.xticks([0, 1], ['apple', 'tomato'])
plt.yticks([0, 1], ['apple', 'tomato'])

# Add number labels to each cell
for i in range(simplified_cm.shape[0]):
    for j in range(simplified_cm.shape[1]):
        plt.text(j, i, format(simplified_cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if simplified_cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('output/apple_vs_tomato_confusion_matrix.png')
plt.close()

# Calculate and save additional metrics
accuracy = accuracy_score(y_true, y_pred)
simplified_accuracy = accuracy_score(simplified_y_true, simplified_y_pred)

# Calculate per-class accuracy
class_accuracy = {}
for i, class_name in enumerate(class_labels):
    # Get indices where true class is this class
    indices = [j for j, val in enumerate(y_true) if val == i]
    if indices:
        # Calculate accuracy for this class
        correct = sum(1 for j in indices if y_pred[j] == y_true[j])
        class_accuracy[class_name] = correct / len(indices)

# Save metrics to JSON
metrics = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "detailed_accuracy": float(accuracy),
    "simplified_accuracy": float(simplified_accuracy),
    "per_class_accuracy": class_accuracy,
    "training_epochs": len(history.history['accuracy']),
    "final_training_accuracy": float(history.history['accuracy'][-1]),
    "final_validation_accuracy": float(history.history['val_accuracy'][-1])
}

with open('output/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Create a metrics summary file
with open('output/metrics_summary.txt', 'w') as f:
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("========================\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(f"Simplified Apple vs Tomato Accuracy: {simplified_accuracy:.4f}\n\n")
    f.write("Per-Class Accuracy:\n")
    for class_name, acc in class_accuracy.items():
        f.write(f"  {class_name}: {acc:.4f}\n")
    f.write("\nTraining completed in {len(history.history['accuracy'])} epochs\n")
    f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")

print("\nTraining and evaluation completed successfully.")
print("Performance metrics and visualizations saved to the output directory.")