from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

train_dir = "working/train"

# Create a data generator
train_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Generate data from the training directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Get the class indices
class_indices = train_generator.class_indices

# Save the class indices to a JSON file
with open("output/class_indices.json", "w") as json_file:
    json.dump(class_indices, json_file)

print("Class Indices:", class_indices)