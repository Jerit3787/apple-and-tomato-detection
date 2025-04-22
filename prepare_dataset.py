import os
import shutil

if not os.path.exists("working"):
    os.mkdir("working")

# Create a array contains name of folder to be copied
data_type = ["Training", "Validation", "Test"]
folders = ["Apple 5", "Apple 7", "Apple 8", "Apple 9", "Apple 10", "Apple 11", "Apple 12", "Apple 13", "Apple 14", "Apple 17", "Apple 18", "Apple 19", "Tomato 1", "Tomato 5", "Tomato 7", "Tomato 8", "Tomato 9", "Tomato 10", "Tomato Cherry Maroon 1", "Tomato Cherry Orange 1", "Tomato Cherry Red 2", "Tomato Cherry Yellow 1", "Tomato Maroon 2"]

# Copy folders from fruits to working
for data in data_type:
    # Create the directory structure for the dataset
    if not os.path.exists(f"working/{data}"):
        os.mkdir(f"working/{data}")

    # Copy images to the respective directories
    for folder in folders:
        full_path = f"fruits-360-original-size/{data}/{folder}/"
        class_count = len(os.listdir(full_path))

        # check if folder exist on destination
        if not os.path.exists(f"working/{data}/{folder}"):
            os.mkdir(f"working/{data}/{folder}")

        # Copy train images
        for img in os.listdir(full_path):
            image_path = full_path + img
            shutil.copy(image_path, f"working/{data}/{folder}")
            print(f"Copying {image_path} to working/{data}/{folder}")
