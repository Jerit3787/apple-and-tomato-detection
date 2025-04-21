import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directories
train_dir = "working/Training"
val_dir = "working/Validation"
test_dir = "working/Test"

# Create a more advanced ImageDataGenerator for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,         # Increased rotation
    width_shift_range=0.25,    # Increased shift
    height_shift_range=0.25,   # Increased shift
    shear_range=0.2,
    zoom_range=0.25,           # Increased zoom
    horizontal_flip=True,
    vertical_flip=False,       # Fruits typically have an up orientation
    brightness_range=[0.7, 1.3], # Add brightness variation
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load training data
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
print("Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print class information and distribution
class_indices = train_generator.class_indices
print("Class indices:", class_indices)
class_counts = {}

for class_name in class_indices.keys():
    class_dir = os.path.join(train_dir, class_name)
    if os.path.exists(class_dir):
        num_images = len(os.listdir(class_dir))
        class_counts[class_name] = num_images
        print(f"Class '{class_name}': {num_images} images")

# Create a deeper CNN model with batch normalization
print("Creating model...")
model = Sequential([
    # First convolutional block
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    # Second convolutional block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    # Third convolutional block
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    # Flatten and dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model with a slightly lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate for better convergence
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

print("Training the model...")

# Train the model with more epochs
steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
validation_steps = max(1, val_generator.samples // val_generator.batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=50,  # Increased epochs since we have early stopping
    callbacks=callbacks,
    verbose=1
)

print("Evaluating the model...")

# Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Don't shuffle for consistent evaluation
)

test_steps = max(1, test_generator.samples // test_generator.batch_size)
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Get predictions for confusion matrix and classification report
test_generator.reset()
y_pred = []
y_true = []

for i in range(test_steps):
    x_batch, y_batch = next(test_generator)
    pred_batch = model.predict(x_batch)
    y_pred.extend(np.argmax(pred_batch, axis=1))
    y_true.extend(np.argmax(y_batch, axis=1))

# Plot confusion matrix and generate classification report
# Get class names from indices
class_names = [k for k, v in sorted(test_generator.class_indices.items(), key=lambda item: item[1])]

# Plot and save confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('output/confusion_matrix.png')

# Generate and save classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)
with open('output/classification_report.txt', 'w') as f:
    f.write(report)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('output/training_history.png')

print("Training completed. Saving results...")

# Save the model
model.save('output/model.keras')

# Save the model architecture and weights
model_json = model.to_json()
with open("output/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("output/model.weights.h5")

# Save the training history
with open("output/history.json", "w") as json_file:
    # Convert numpy values to Python types for JSON serialization
    history_dict = {}
    for key, value in history.history.items():
        history_dict[key] = [float(x) for x in value]
    json.dump(history_dict, json_file)

# Save the class indices
with open("output/class_indices.json", "w") as json_file:
    json.dump(train_generator.class_indices, json_file)

# Save the model summary to a text file
with open("output/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Create simplified class mapping
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

# Save generator configurations
generator_config = {
    'train': {
        'directory': train_dir,
        'target_size': (224, 224),
        'batch_size': 32,
        'class_mode': 'categorical',
        'preprocessing_config': {
            'rescale': 1.0/255.0,
            'rotation_range': 30,
            'width_shift_range': 0.25,
            'height_shift_range': 0.25,
            'shear_range': 0.2,
            'zoom_range': 0.25,
            'horizontal_flip': True,
            'brightness_range': [0.7, 1.3],
            'fill_mode': 'nearest'
        }
    },
    'validation': {
        'directory': val_dir,
        'target_size': (224, 224),
        'batch_size': 32,
        'class_mode': 'categorical',
        'preprocessing_config': {
            'rescale': 1.0/255.0
        }
    },
    'test': {
        'directory': test_dir,
        'target_size': (224, 224),
        'batch_size': 32,
        'class_mode': 'categorical',
        'preprocessing_config': {
            'rescale': 1.0/255.0
        }
    }
}

with open("output/generator_configs.json", "w") as f:
    json.dump(generator_config, f)

# Save sample data for troubleshooting (safely)
try:
    train_generator.reset()
    val_generator.reset()
    test_generator.reset()
    
    # Get one batch from each generator
    train_batch = next(train_generator)
    val_batch = next(val_generator)
    test_batch = next(test_generator)
    
    # Save safely with handling for empty generators
    if len(train_batch) > 1 and train_batch[0].size > 0:
        with open("output/train_data_sample.npy", "wb") as f:
            np.save(f, train_batch[0][:5])  # Save just 5 samples to save space
        with open("output/train_labels_sample.npy", "wb") as f:
            np.save(f, train_batch[1][:5])
    
    if len(val_batch) > 1 and val_batch[0].size > 0:
        with open("output/val_data_sample.npy", "wb") as f:
            np.save(f, val_batch[0][:5])
        with open("output/val_labels_sample.npy", "wb") as f:
            np.save(f, val_batch[1][:5])
    
    if len(test_batch) > 1 and test_batch[0].size > 0:
        with open("output/test_data_sample.npy", "wb") as f:
            np.save(f, test_batch[0][:5])
        with open("output/test_labels_sample.npy", "wb") as f:
            np.save(f, test_batch[1][:5])
except Exception as e:
    print(f"Warning: Could not save sample data: {e}")

print("Training results saved successfully.")