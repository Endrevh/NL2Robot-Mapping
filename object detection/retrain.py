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

    print("Training for epochs=", epochs, ", learning_rate=", learning_rate)

    # Train the model
    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options)

    return model

def loop_over_hyperparameters(train_data, validation_data, epochs_list, learning_rates, results_file):
    # If the file does not exist, create it as an empty file
    with open(results_file, "a") as f:
        f.write("")

    # Loop over the different combinations of hyperparameters and save results
    for epochs in epochs_list:
        for learning_rate in learning_rates:
            # Read from training_results.txt to avoid duplicate training
            with open(results_file, "r") as f:
                skip = False
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line == "":
                        continue
                    epoch = line.split(",")[0]
                    lr = line.split(",")[1]
                    if int(epoch) == epochs and float(lr) == learning_rate:
                        skip = True
                        break
                # Skip training if already trained
                if skip:
                    print("Skipped training for this combination of hyperparameters: epochs=", epochs, ", learning_rate=", learning_rate)
                    continue

            model = retrain(train_data, validation_data, epochs=epochs, batch_size=8, learning_rate=learning_rate)
            loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
            with open("training_results.txt", "a") as f:
                f.write(f"{epochs},{learning_rate},{coco_metrics['AP']},{loss}\n")

# Load the datasets
train_dataset_path = "train"
validation_dataset_path = "val"
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")

# Select hyperparameters for loop_over_hyperparameters
epochs = [10, 20, 30, 40, 50, 60]
learning_rates = [0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50]

# Measure time taken to train the model
start = time.time()

# Loop over hyperparameters
results_file = "training_results.txt"
loop_over_hyperparameters(train_data, validation_data, epochs, learning_rates, results_file)

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
