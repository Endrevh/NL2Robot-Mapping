import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    max_samples=500,
    seed=52,
    classes=["Screwdriver"],
    shuffle=True,
)

session = fo.launch_app(dataset.view())
session.view = dataset.view()

session.wait()
