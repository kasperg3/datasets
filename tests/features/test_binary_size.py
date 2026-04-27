import pyarrow as pa
import pytest

from datasets import Audio, Image, Nifti, Pdf, Video
from datasets.features import audio as audio_module
from datasets.features import image as image_module
from datasets.features import video as video_module


@pytest.mark.parametrize(
    "feature_name, feature",
    [
        ("Image", Image()),
        ("Audio", Audio()),
        ("Video", Video()),
        ("Pdf", Pdf()),
        ("Nifti", Nifti()),
    ],
)
def test_embed_storage_raises_clear_overflow(monkeypatch, feature_name, feature):
    monkeypatch.setattr("datasets.features._binary_size._PA_BINARY_MAX_BYTES", 2)
    storage = pa.array(
        [{"bytes": b"abc", "path": None}],
        type=pa.struct({"bytes": pa.binary(), "path": pa.string()}),
    )

    with pytest.raises(OverflowError, match=rf"\[{feature_name}\] cannot fit 3 bytes"):
        feature.embed_storage(storage)


@pytest.mark.parametrize(
    "feature_name, feature, module",
    [
        ("Image", Image(), image_module),
        ("Audio", Audio(), audio_module),
        ("Video", Video(), video_module),
    ],
)
def test_large_binary_cast_is_wrapped(monkeypatch, feature_name, feature, module):
    def raise_arrow_invalid(*args, **kwargs):
        raise pa.ArrowInvalid("overflow")

    monkeypatch.setattr(module, "array_cast", raise_arrow_invalid)
    storage = pa.array([b"abc"], type=pa.large_binary())

    with pytest.raises(OverflowError, match=rf"\[{feature_name}\] failed to downcast large_binary -> binary"):
        feature.cast_storage(storage)
