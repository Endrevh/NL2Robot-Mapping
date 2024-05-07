import tensorflow as tf
import time
assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector

def retrain(train_data, validation_data, epochs, batch_size, learning_rate):
    # Hyperparameters
    spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG_I384
    hparams = object_detector.HParams(export_dir='exported_model', epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams
    )

    # Train the model
    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options)

    return model

# Load the datasets
train_dataset_path = "train"
validation_dataset_path = "val"
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")

# Measure time taken to train the model
start = time.time()

# Loop over the different combinations of hyperparameters and save results
for epochs in range(10, 110, 10):
    for learning_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        model = retrain(train_data, validation_data, epochs=epochs, batch_size=8, learning_rate=learning_rate)
        loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
        with open("training_results.txt", "a") as f:
            f.write(f"{epochs},{learning_rate},{coco_metrics['AP']}\n")

end = time.time()
print(f"Training took {end - start} seconds")

# Retrain the model
# model = retrain(train_data, validation_data, epochs=10, batch_size=8, learning_rate=0.01)

# Evaluate the model
#loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
#print(f"Validation loss: {loss}")
#print(f"Validation coco metrics: {coco_metrics}")

# Export the model
#model.export_model()
