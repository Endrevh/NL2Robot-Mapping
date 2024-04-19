import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector

# Load the dataset
train_dataset_path = "hammer_train"
validation_dataset_path = "hammer_val"
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)

# Hyperparameters
spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG_I384
hparams = object_detector.HParams(export_dir='exported_model')
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)

# Train the model
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)

# Evaluate the model
#loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
#print(f"Validation loss: {loss}")
#print(f"Validation coco metrics: {coco_metrics}")

# Export the model
model.export_model()
