import numpy as numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

train_dir = "working/train"
val_dir = "working/val"
test_dir = "working/test"

# Create ImageDataGenerator for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load training data

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224), # Resize images to 224x224
    batch_size=32,
    class_mode='categorical' # Use 'categorical' for multi-class classification
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax') # Number of classes
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20 # Adjust the number of epochs as needed
)

# Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

import os

os.mkdir("output")

# Save the model
model.save('output/model.keras')

# Save the model architecture and weights
model_json = model.to_json()
with open("output/model.json", "w") as json_file:
    json_file.write(model_json)
with open("output/model.weights.h5", "wb") as h5_file:
    model.save_weights(h5_file)

# Save the training history
import json
with open("output/history.json", "w") as json_file:
    json.dump(history.history, json_file)

# Save the class indices
import json
with open("output/class_indices.json", "w") as json_file:
    json.dump(train_generator.class_indices, json_file)

# Save the model summary to a text file
with open("output/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Save the training and validation data generators to a file
import pickle
with open("output/train_generator.pkl", "wb") as f:
    pickle.dump(train_generator, f)
with open("output/val_generator.pkl", "wb") as f:
    pickle.dump(val_generator, f)

# Save the test data generator to a file
with open("output/test_generator.pkl", "wb") as f:
    pickle.dump(test_generator, f)

# Save the training and validation data to a file
import numpy as np
with open("output/train_data.npy", "wb") as f:
    np.save(f, train_generator[0][0])
with open("output/val_data.npy", "wb") as f:
    np.save(f, val_generator[0][0])
with open("output/test_data.npy", "wb") as f:
    np.save(f, test_generator[0][0])

# Save the training and validation labels to a file
with open("output/train_labels.npy", "wb") as f:
    np.save(f, train_generator[0][1])
with open("output/val_labels.npy", "wb") as f:
    np.save(f, val_generator[0][1])
with open("output/test_labels.npy", "wb") as f:
    np.save(f, test_generator[0][1])

