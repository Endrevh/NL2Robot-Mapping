import fiftyone as fo
import fiftyone.utils.random as four
import os
import shutil

# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type = fo.types.COCODetectionDataset,
    data_path = "hammer_export/images",
    labels_path = "hammer_export/result.json",
    include_id = True,
)

# Verify that the class list for our dataset was imported
object_categories = coco_dataset.default_classes
print(object_categories)

# Split the dataset randomly into train, validation, and test sets
four.random_split(
    coco_dataset,
    {"train": 0.8, "val": 0.1, "test": 0.1},
)
train_view = coco_dataset.match_tags("train")
val_view = coco_dataset.match_tags("val")
test_view = coco_dataset.match_tags("test")

# Export the datasets, clean directories if they already exist
train_export_dir = "hammer_train"
val_export_dir = "hammer_val"
test_export_dir = "hammer_test"
if os.path.exists(train_export_dir):
    shutil.rmtree(train_export_dir)
if os.path.exists(val_export_dir):
    shutil.rmtree(val_export_dir)
if os.path.exists(test_export_dir):
    shutil.rmtree(test_export_dir)
train_view.export(export_dir=train_export_dir, dataset_type=fo.types.COCODetectionDataset)
val_view.export(export_dir=val_export_dir, dataset_type=fo.types.COCODetectionDataset)
test_view.export(export_dir=test_export_dir, dataset_type=fo.types.COCODetectionDataset)

# Print the detected object categories
detected_object_categories = train_view.default_classes
print("Detected object categories: ", detected_object_categories)

# Adjust json-files so that background has ID 0, and the rest of the classes are shifted by 1. All occurances of the labels in the json-files are adjusted accordingly.
# Add background, and shift all other classes by 1. Then, find and replace all occurances, start from the back
def adjust_json_file(json_file_path, object_categories):
    n = len(object_categories)
    with open(json_file_path, "r") as file:
        data = file.read()
        for i in range(n, 0, -1):
            data = data.replace(f'"category_id": {i-1}', f'"category_id": {i}')
            data = data.replace(f'"name": "{object_categories[i-1]}"', f'"name": "{object_categories[i-1]}"')

def adjust_json_file(json_file_path, object_categories):
    with open(json_file_path, "r") as file:
        data = file.readlines()

    for i, line in enumerate(data):
        for j, category in enumerate(object_categories):
            if category in line:
                data[i] = line.replace(f'"id": {j+1}', f'"id": {j+1}')
                data[i] = data[i].replace(f'"name": "{category}"', f'"name": "{category}"')
                data[i] = data[i].replace(f'"id": {j}', f'"id": {j+1}')
                data[i] = data[i].replace(f'"name": "{category}"', f'"name": "{category}"')

    with open(json_file_path, "w") as file:
        file.writelines(data)

# Check how many images from each class are in each dataset
class_counts_train = {}
for file in os.listdir(train_export_dir+"\data"):
    filename = os.fsdecode(file)
    for category in object_categories:
        if category.lower() in filename:
            class_counts_train[category] = class_counts_train.get(category, 0) + 1

class_counts_val = {}
for file in os.listdir(val_export_dir+"\data"):
    filename = os.fsdecode(file)
    for category in object_categories:
        if category.lower() in filename:
            class_counts_val[category] = class_counts_val.get(category, 0) + 1

class_counts_test = {}
for file in os.listdir(test_export_dir+"\data"):
    filename = os.fsdecode(file)
    for category in object_categories:
        if category.lower() in filename:
            class_counts_test[category] = class_counts_test.get(category, 0) + 1

# Print the counts for each class
for category in class_counts_train.keys():
    print(f"{category}: {class_counts_train[category]} in train, {class_counts_val[category]} in val, {class_counts_test[category]} in test")

# Create list of detected object categories in the train dataset
detected_object_categories_train = []
for file in os.listdir(train_export_dir+"\data"):
    filename = os.fsdecode(file)
    for category in object_categories:
        if category in filename and category not in detected_object_categories_train:
            detected_object_categories_train.append(category)
print("Detected object categories: ", detected_object_categories_train)