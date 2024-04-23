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
print(coco_dataset.default_classes) # ['hammer', ...]

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
train_view.export(export_dir=train_export_dir, dataset_type=fo.types.COCODetectionDataset)
val_view.export(export_dir=val_export_dir, dataset_type=fo.types.COCODetectionDataset)
if os.path.exists(train_export_dir):
    shutil.rmtree(train_export_dir)
if os.path.exists(val_export_dir):
    shutil.rmtree(val_export_dir)
if os.path.exists(test_export_dir):
    shutil.rmtree(test_export_dir)
test_view.export(export_dir=test_export_dir, dataset_type=fo.types.COCODetectionDataset)
