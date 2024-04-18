import fiftyone as fo
import fiftyone.utils.random as four

# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type = fo.types.COCODetectionDataset,
    data_path = "hammer_dataset/images",
    labels_path = "hammer_dataset/result.json",
    include_id = True,
)

# Verify that the class list for our dataset was imported
print(coco_dataset.default_classes) # ['hammer', ...]

print(coco_dataset)

# Split the dataset randomly into train, validation, and test sets
four.random_split(
    coco_dataset,
    {"train": 0.8, "validation": 0.1, "test": 0.1},
)

train_view = coco_dataset.match_tags("train")
test_view = coco_dataset.match_tags("test")
val_view = coco_dataset.match_tags("val")

# Export the datasets

# train_view.export(export_dir=train_export_dir, dataset_type=fo.types.COCODetectionDataset)
