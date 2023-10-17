import os
import sys
import tempfile
from typing import Any

import datasets
import numpy as np
from absl import app
from absl import flags
from PIL import Image
from tqdm.auto import tqdm


FLAGS = flags.FLAGS


flags.DEFINE_string("output-dir", os.getcwd(), "The DIBCO dataset dir")
flags.DEFINE_integer("seed", 0, "The random state seed")
flags.DEFINE_float("eval-size", 0.2, "The eval split size")
flags.DEFINE_float("test-size", 0.2, "The test split size")


def normalize(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    
    array = np.array(image).astype(np.uint8)
    condition = array < np.max(array)
    array = np.where(condition, 1, 0).astype(bool)
    
    image = Image.fromarray(array)
    return image


def preprocessing(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """It prepares a batch of examples for semantic segmentation"""
    sources = batch["source"]
    targets = batch["target"]

    batch = {
        "labelmap": [normalize(Image.open(src)) for src in targets],
        "pixelmap": [Image.open(src) for src in sources]
    }

    return batch


def main(argv):
    del argv

    URL = "https://github.com/Leedeng/SauvolaNet.git"

    assert FLAGS.eval_size + FLAGS.test_size < 1.0, "The splits must sum to less than 1.0"

    with tempfile.TemporaryDirectory() as sauvolanet:
        os.system(f"git clone {URL} {sauvolanet}")

        dataset_dir = os.path.join(sauvolanet, "Dataset")
        src_dir = os.path.join(sauvolanet, "SauvolaDocBin")

        sys.path.insert(0, src_dir)

        from dataUtils import collect_binarization_by_dataset
        collection = collect_binarization_by_dataset(dataset_dir)
        del collect_binarization_by_dataset

        sys.path.remove(src_dir)

        del src_dir
        del dataset_dir

        features = datasets.Features({
            "ensemble": datasets.Value("string"),
            "source": datasets.Value("string"),
            "target": datasets.Value("string"),
        })

        for key, examples in tqdm(collection.items(), desc="DIBCO benchmark"):
            sources, targets = zip(*examples)

            sources = sorted(sources)
            targets = sorted(targets)

            dataset = {"source": sources, "target": targets, "ensemble": [key] * len(sources)}
            dataset = datasets.Dataset.from_dict(dataset, features)

            collection[key] = dataset

        collection = datasets.concatenate_datasets([
            dataset for _, dataset in collection.items()
        ])

        features = datasets.Features({
            "ensemble": datasets.Value("string"),
            "labelmap": datasets.Image(),
            "pixelmap": datasets.Image(),
        })

        collection = collection.map(
            preprocessing, 
            batched=True,
            features=features, 
            remove_columns=["source", "target"]
        )

        collection = collection.class_encode_column("ensemble")

        train_size = 1.0 - FLAGS.test_size

        collection = collection.train_test_split(
            seed=FLAGS.seed,
            shuffle=True,
            stratify_by_column="ensemble",
            train_size=train_size
        )

        collection.save_to_disk(FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
