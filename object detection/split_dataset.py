# This file takes as input the exported dataset from Label Studio and splits it into train, validation, and test sets, all on COCO format.

import fiftyone as fo
import fiftyone.utils.random as four
import os
import shutil
import re

# Add license to json-file
def add_license_to_json_file(json_file_path, license_name, license_url):
    with open(json_file_path, "r") as file:
        data = file.read()
        replace_text = '"licenses": []'
        new_license = '"licenses": [{"url": "'+license_url+'","name": "'+license_name+'"}]'
        data = data.replace(replace_text, new_license)
    with open(json_file_path, "w") as file:
        file.write(data)

# Adjust json-files so that background has ID 0, and the rest of the classes are shifted by 1
def adjust_IDs_in_json_file(json_file_path, object_categories):
    n = len(object_categories)
    with open(json_file_path, "r") as file:
        data = file.read()
        # Shift all categories by 1
        for i in range(n, 0, -1):
            data = data.replace(f'"id": {i-1}', f'"id": {i}')
            data = data.replace(f'"category_id": {i-1}', f'"category_id": {i}')

        # Get the categories part of the json-file
        replace = re.search('"categories": (.*),"images":', data).group(1)
        
        # Add background category as ID 0 at the start
        new = f'[{{"id": 0,"name": "background"}},{replace[1:]}'
        
        # Replace the categories part of the json-file
        data = data.replace(replace, new)

    # Finally, write the updated json-file
    with open(json_file_path, "w") as file:
        file.write(data)

# First, remove unnecessary part of image paths in the result.json file
label_studio_export_dir = "LS_export"
with open(label_studio_export_dir+"/result.json", "r") as file:
    data = file.read()
    data = data.replace('"file_name": "images\\\\', '"file_name": "')
with open(label_studio_export_dir+"/result.json", "w") as file:
    file.write(data)

# Load COCO formatted dataset
print("Loading dataset from Label Studio export...")
coco_dataset = fo.Dataset.from_dir(
    dataset_type = fo.types.COCODetectionDataset,
    data_path = label_studio_export_dir+"/images",
    labels_path = label_studio_export_dir+"/result.json",
    include_id = True,
)

# Verify that the class list for our dataset was imported
object_categories = coco_dataset.default_classes
print("Detected object categories in Label Studio export:", object_categories)

# Split the dataset randomly into train, validation, and test sets
four.random_split(
    coco_dataset,
    {"train": 0.8, "val": 0.1, "test": 0.1},
)
train_view = coco_dataset.match_tags("train")
val_view = coco_dataset.match_tags("val")
test_view = coco_dataset.match_tags("test")

# Export the datasets, clean directories if they already exist
train_export_dir = "train"
val_export_dir = "val"
test_export_dir = "test"
if os.path.exists(train_export_dir):
    shutil.rmtree(train_export_dir)
if os.path.exists(val_export_dir):
    shutil.rmtree(val_export_dir)
if os.path.exists(test_export_dir):
    shutil.rmtree(test_export_dir)

print("Exporting train dataset to COCO format...")
train_view.export(export_dir=train_export_dir, dataset_type=fo.types.COCODetectionDataset)
print("Exporting validation dataset to COCO format...")
val_view.export(export_dir=val_export_dir, dataset_type=fo.types.COCODetectionDataset)
print("Exporting test dataset to COCO format...")
test_view.export(export_dir=test_export_dir, dataset_type=fo.types.COCODetectionDataset)

# Check how many images from each class are in each dataset
class_counts_train = {}
for file in os.listdir(train_export_dir+"/data"):
    filename = os.fsdecode(file)
    for category in object_categories:
        if category.lower() in filename:
            class_counts_train[category] = class_counts_train.get(category, 0) + 1

class_counts_val = {}
for file in os.listdir(val_export_dir+"/data"):
    filename = os.fsdecode(file)
    for category in object_categories:
        if category.lower() in filename:
            class_counts_val[category] = class_counts_val.get(category, 0) + 1

class_counts_test = {}
for file in os.listdir(test_export_dir+"/data"):
    filename = os.fsdecode(file)
    for category in object_categories:
        if category.lower() in filename:
            class_counts_test[category] = class_counts_test.get(category, 0) + 1

# Print the counts for each class
print("\n\nNumber of images for each class in each dataset:")
for category in class_counts_train.keys():
    print(f"{category}: {class_counts_train[category]} in train, {class_counts_val[category]} in val, {class_counts_test[category]} in test")

# Add license to json-files
license_name = "Attribution-ShareAlike 4.0 International"
license_url = "https://creativecommons.org/licenses/by-sa/4.0/"
add_license_to_json_file(train_export_dir+"/labels.json", license_name, license_url)
add_license_to_json_file(val_export_dir+"/labels.json", license_name, license_url)
add_license_to_json_file(test_export_dir+"/labels.json", license_name, license_url)

# Adjust IDs in json-files
adjust_IDs_in_json_file(train_export_dir+"/labels.json", object_categories)
adjust_IDs_in_json_file(val_export_dir+"/labels.json", object_categories)
adjust_IDs_in_json_file(test_export_dir+"/labels.json", object_categories)