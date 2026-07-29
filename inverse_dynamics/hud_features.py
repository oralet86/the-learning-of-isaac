"""Forward-fill helper for per-chunk HUD feature arrays.

Operates one stage earlier than ``inverse_dynamics/transforms.py``'s
mask functions — on the raw HUD array produced by
``components.hud_ocr.TorchHudExtractor`` inside ``chunk_video.py``,
before ``preprocess.py`` even runs — so it lives in its own module
rather than being folded into ``transforms.py``.
"""

import numpy as np


def forward_fill_hud(
    hud: np.ndarray, missing_value: float = float("nan")
) -> np.ndarray:
    """Forward-fill missing values per column from the last valid value.

    No state persists across calls: a chunk's leading frames are left
    at *missing_value* if no valid value has occurred yet within this
    array, matching the pipeline's per-chunk independence (memory usage
    must not scale with chunk count, and chunks may span different
    recording runs).

    Args:
        hud: ``(N, H_DIM)`` float array. Entries equal to
            *missing_value* (compared via ``np.isnan`` when
            *missing_value* is NaN, else exact equality) are filled from
            the most recent non-missing value in the same column.
        missing_value: Sentinel marking an unreadable field.

    Returns:
        ``(N, H_DIM)`` array, forward-filled in place of missing
        entries. A no-op on columns that contain no missing values.
    """
    if np.isnan(missing_value):
        is_missing = np.isnan(hud)
    else:
        is_missing = hud == missing_value

    valid = ~is_missing
    n = hud.shape[0]
    last_valid_idx = np.where(valid, np.arange(n)[:, None], 0)
    np.maximum.accumulate(last_valid_idx, axis=0, out=last_valid_idx)

    cols = np.arange(hud.shape[1])[None, :]
    filled = hud[last_valid_idx, cols]

    never_valid = ~np.maximum.accumulate(valid, axis=0)
    filled[never_valid] = missing_value
    return filled
