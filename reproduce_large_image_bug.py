"""End-to-end repro for the Image.save_to_disk bug.

This script uses plain random PIL images and exercises Dataset.save_to_disk()
directly. It does not monkeypatch datasets internals.

The dataset is intentionally large enough to hit the buggy code path in the
older datasets commit once the dependency is pinned there.
"""

from __future__ import annotations

import tempfile

import numpy as np
from PIL import Image as PILImage

from datasets import Dataset, Features, Image, Sequence, Value


def random_pil_image(rng: np.random.Generator, size: int = 4096) -> PILImage.Image:
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return PILImage.fromarray(array, mode="RGB")


def build_examples(count: int = 100) -> list[dict[str, object]]:
    rng = np.random.default_rng(4)
    examples: list[dict[str, object]] = []
    for index in range(count):
        examples.append(
            {
                "id": str(index),
                "bracket": [
                    random_pil_image(rng),
                    random_pil_image(rng),
                    random_pil_image(rng),
                    random_pil_image(rng),
                    random_pil_image(rng),
                    random_pil_image(rng),
                    random_pil_image(rng),
                    random_pil_image(rng),
                ],
                "enhanced1": random_pil_image(rng),
                "enhanced2": random_pil_image(rng),
                "enhanced3": random_pil_image(rng),
                "enhanced4": random_pil_image(rng),
                "enhanced5": random_pil_image(rng),
                "enhanced6": random_pil_image(rng),
            }
        )
    return examples


def build_dataset() -> Dataset:
    features = Features(
        {
            "id": Value("string"),
            "bracket": Sequence(Image()),
            "enhanced1": Image(),
            "enhanced2": Image(),
            "enhanced3": Image(),
            "enhanced4": Image(),
            "enhanced5": Image(),
            "enhanced6": Image(),
        }
    )
    return Dataset.from_list(build_examples(), features=features)


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = f"{temp_dir}/dataset"
        print("Building dataset...")
        dataset = build_dataset()
        print("Dataset built with", len(dataset), "rows")
        print("Calling save_to_disk()...")
        dataset.save_to_disk(output_dir)
        print("save_to_disk() completed")
    return 0


if __name__ == "__main__":
    main()
