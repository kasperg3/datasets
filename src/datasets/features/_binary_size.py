_PA_BINARY_MAX_BYTES = (1 << 31) - 1


def check_pa_binary_fits(byte_objects, feature_name: str) -> None:
    """Raise OverflowError if the bytes would not fit in a single pa.binary() array."""
    total = 0
    n_non_null = 0
    largest = 0
    for b in byte_objects:
        if b is None:
            continue
        size = len(b)
        total += size
        n_non_null += 1
        if size > largest:
            largest = size
    if total > _PA_BINARY_MAX_BYTES:
        avg = total // n_non_null if n_non_null else 0
        raise OverflowError(
            f"[{feature_name}] cannot fit {total:,} bytes across {n_non_null} non-null row(s) into a single "
            f"pa.binary() array (limit {_PA_BINARY_MAX_BYTES:,} bytes / 2 GiB; largest row = {largest:,} bytes, "
            f"average = {avg:,} bytes). Reduce writer_batch_size so each batch stays under 2 GiB, or keep paths "
            f"instead of embedding bytes (do not call save_to_disk/push_to_hub after cast_column to bytes for very "
            f"large media)."
        )
