"""Signed HUD-diff transform — encodes HUD change as deltas around the label frame."""

import torch
import torch.nn as nn


class HudDiff(nn.Module):
    """Encode a 3-frame HUD window ``(t-1, t, t+1)`` as two signed deltas.

    Input:  ``(B, 3, H_DIM)`` float — raw HUD vectors at ``t-1``, ``t``, ``t+1``.
    Output: ``(B, 2, H_DIM)`` float — ``[hud[t] - hud[t-1], hud[t+1] - hud[t]]``.

    Unlike ``SignedDiff``, no ``[0, 1]`` clamping/centering is applied —
    that convention is specific to normalized pixel values, whereas raw
    HUD fields (stat values, heart-slot ids, coin counts, etc.) aren't
    normalized.
    """

    def forward(self, hud: torch.Tensor) -> torch.Tensor:
        return hud[:, 1:] - hud[:, :-1]
