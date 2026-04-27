import pyarrow as pa


_OVERFLOW_MSG = (
    "[{feature_name}] embedded bytes exceed the 2 GiB pa.binary() limit for a single "
    "Arrow array. Reduce `writer_batch_size` so each batch stays under 2 GiB, or keep "
    "paths instead of embedding bytes for very large media."
)


def binary_array_or_overflow(values, feature_name: str) -> pa.Array:
    """Build a `pa.binary()` array from values, or raise `OverflowError` when the total
    bytes exceed the 2 GiB int32-offset limit of `pa.binary()`.

    Two failure modes are normalized into the same `OverflowError`:
    - Older PyArrow: `pa.array(..., type=pa.binary())` raises `pa.ArrowInvalid`.
    - Newer PyArrow: it returns a `pa.ChunkedArray`, which would break the downstream
      `pa.StructArray.from_arrays(..., mask=...)` call (issue #5717).
    """
    try:
        arr = pa.array(values, type=pa.binary())
    except pa.ArrowInvalid as e:
        raise OverflowError(_OVERFLOW_MSG.format(feature_name=feature_name)) from e
    if isinstance(arr, pa.ChunkedArray):
        raise OverflowError(_OVERFLOW_MSG.format(feature_name=feature_name))
    return arr
