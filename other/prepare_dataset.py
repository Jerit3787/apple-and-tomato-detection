import os
import shutil
data_path = "train/"

# Create the directory structure for the dataset
if not os.path.exists("working"):
    os.mkdir("working")
    os.mkdir("working/train")
    os.mkdir("working/train/class-1")
    os.mkdir("working/train/class-2")
    os.mkdir("working/train/class-3")
    os.mkdir("working/train/class-4")
    os.mkdir("working/train/class-5")
    os.mkdir("working/val")
    os.mkdir("working/val/class-1")
    os.mkdir("working/val/class-2")
    os.mkdir("working/val/class-3")
    os.mkdir("working/val/class-4")
    os.mkdir("working/val/class-5")
    os.mkdir("working/test")
    os.mkdir("working/test/class-1")
    os.mkdir("working/test/class-2")
    os.mkdir("working/test/class-3")
    os.mkdir("working/test/class-4")
    os.mkdir("working/test/class-5")

# Copy images to the respective directories
for class_path in os.listdir(data_path):
    full_path = data_path + class_path + "/"
    class_count = len(os.listdir(data_path + class_path))
    train_count = int(class_count * 0.7)
    val_count = int(class_count * 0.1)

    # Copy train images
    for img in os.listdir(full_path)[0:train_count]:
        image_path = full_path + img
        shutil.copy(image_path, f"working/train/{class_path}")
        print(f"Copying {image_path} to working/train/{class_path}")

    # Copy val images
    for img in os.listdir(full_path)[train_count:(train_count + val_count)]:
        image_path = full_path + img
        shutil.copy(image_path, f"working/val/{class_path}")
        print(f"Copying {image_path} to working/val/{class_path}")

    # Copy test images
    for img in os.listdir(full_path)[(train_count + val_count):-1]:
        image_path = full_path + img
        shutil.copy(image_path, f"working/test/{class_path}")
        print(f"Copying {image_path} to working/test/{class_path}")

print("Dataset preparation complete.")