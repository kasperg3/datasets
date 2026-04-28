import resource
import threading
import timeit

import numpy as np

import datasets
from datasets.arrow_writer import ArrowWriter
from datasets.features.features import _ArrayXD


def get_duration(func):
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        _ = func(*args, **kwargs)
        delta = timeit.default_timer() - starttime
        return delta

    wrapper.__name__ = func.__name__

    return wrapper


def rss_mb() -> float:
    """Instantaneous RSS (MB) from /proc/self/status — Linux only."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def maxrss_mb() -> float:
    """Process high-water RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class RSSPoller(threading.Thread):
    """Background thread that polls RSS and records the peak."""

    def __init__(self, interval: float = 0.01):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            r = rss_mb()
            if r > self.peak:
                self.peak = r
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join()


def generate_examples(features: dict, num_examples=100, seq_shapes=None):
    dummy_data = []
    seq_shapes = seq_shapes or {}
    for i in range(num_examples):
        example = {}
        for col_id, (k, v) in enumerate(features.items()):
            if isinstance(v, _ArrayXD):
                data = np.random.rand(*v.shape).astype(v.dtype)
            elif isinstance(v, datasets.Value):
                if v.dtype == "string":
                    data = "The small grey turtle was surprisingly fast when challenged."
                else:
                    data = np.random.randint(10, size=1).astype(v.dtype).item()
            elif isinstance(v, datasets.Sequence):
                while isinstance(v, datasets.Sequence):
                    v = v.feature
                shape = seq_shapes[k]
                data = np.random.rand(*shape).astype(v.dtype)
            example[k] = data

        dummy_data.append((i, example))

    return dummy_data


def generate_example_dataset(dataset_path, features, num_examples=100, seq_shapes=None):
    dummy_data = generate_examples(features, num_examples=num_examples, seq_shapes=seq_shapes)

    with ArrowWriter(features=features, path=dataset_path) as writer:
        for key, record in dummy_data:
            example = features.encode_example(record)
            writer.write(example)

        num_final_examples, num_bytes = writer.finalize()

    if not num_final_examples == num_examples:
        raise ValueError(
            f"Error writing the dataset, wrote {num_final_examples} examples but should have written {num_examples}."
        )

    dataset = datasets.Dataset.from_file(filename=dataset_path, info=datasets.DatasetInfo(features=features))

    return dataset
