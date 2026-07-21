"""Signed-diff transform — encodes motion as per-frame brightness deltas.

Frame 0 is kept as raw grayscale.  Frames 1..T-1 are replaced by the
clamped signed difference from the previous frame, centred at 0.5.
"""

import torch
import torch.nn as nn


class SignedDiff(nn.Module):
    """Apply signed frame differences as a tensor operation.

    Input:  ``(B, T, H, W)`` float tensor in ``[0, 1]``.
    Output: ``(B, T, H, W)`` float tensor in ``[0, 1]``.

    Values > 0.5 mean the pixel got brighter; < 0.5 means darker.
    Equivalent to the uint8 centre-at-128 convention after normalisation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        if t > 1:
            frame0 = x[:, :1]
            diffs = x[:, 1:] - x[:, :-1]
            x = torch.cat([frame0, (0.5 + diffs).clamp(0, 1)], dim=1)
        return x
