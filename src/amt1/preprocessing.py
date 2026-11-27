# SPDX-License-Identifier: Apache-2.0
"""Lossless RGB adaptation for color-based foreground estimation."""
from io import BytesIO

import numpy as np
from PIL import Image


def colour_foreground(image: Image.Image) -> np.ndarray:
    """Compute a color-based foreground mask from unlabeled RGB pixels.

    Args:
        image: Pillow image, converted to RGB without resizing before the helper.

    Returns:
        Boolean array with shape (784,) on the 28-by-28 patch grid.
    """
    from ._vendor.radiology.foreground import _colour_fg

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return _colour_fg(buffer, (28, 28))[0]
