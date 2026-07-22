"""Image Quality Validation Module for GenX 360 AI Service.

Validates input image quality before passing to ONNX model inference.
Flags unanalyzable garbage inputs (solid colors, pitch dark, overexposed, severely blurry images).
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image


def validate_image_quality(image_bytes: bytes) -> tuple[bool, str | None]:
    """Validate image quality.

    Returns:
        tuple[bool, str | None]: (is_valid, failure_reason)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False, "Corrupted or invalid image file format"

    width, height = img.size
    if width < 64 or height < 64:
        return False, f"Image resolution too low ({width}x{height}); minimum 64x64 required"

    img_arr = np.array(img, dtype=np.float32)

    # 1. Solid / Low Variance Color Check
    std_per_channel = np.std(img_arr, axis=(0, 1))
    overall_std = float(np.mean(std_per_channel))
    if overall_std < 10.0:
        return False, f"Image is solid/near-solid color (low pixel variance: {overall_std:.1f})"

    # 2. Extreme Brightness / Darkness Check
    mean_val = float(np.mean(img_arr))
    if mean_val < 12.0:
        return False, f"Image is too dark to analyze (mean brightness: {mean_val:.1f})"
    if mean_val > 243.0:
        return False, f"Image is overexposed (mean brightness: {mean_val:.1f})"

    # 3. Blur Detection (High-frequency gradient variance check)
    gray = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
    # Simple 2D Laplacian kernel gradient approximation
    laplacian = (
        np.abs(np.roll(gray, 1, axis=0) - gray) +
        np.abs(np.roll(gray, -1, axis=0) - gray) +
        np.abs(np.roll(gray, 1, axis=1) - gray) +
        np.abs(np.roll(gray, -1, axis=1) - gray)
    )
    blur_score = float(np.var(laplacian))
    if blur_score < 30.0:
        return False, f"Image is severely blurry (blur score: {blur_score:.1f})"

    return True, None
