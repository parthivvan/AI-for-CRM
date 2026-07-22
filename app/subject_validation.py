"""Subject Validation & Non-Clinical Content Detector for GenX 360 AI.

Verifies whether an uploaded image contains relevant human clinical content 
(Skin, Scalp, Body) matching the requested ImageType.

Rejects unrelated, non-clinical, or fake uploads (e.g., objects, landscapes,
documents, animals) and instructs the user to upload only related clinical images.
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image
from app.schemas import ImageType


def validate_subject_relevance(
    image_bytes: bytes,
    image_type: ImageType,
    min_skin_hair_ratio: float = 0.12,
) -> tuple[bool, str | None]:
    """Verify image contains relevant human skin/scalp/body content.

    Returns:
        tuple[bool, str | None]: (is_valid, user_facing_error_message)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False, "Corrupted or invalid image file. Please upload a valid JPEG/PNG image."

    # Convert RGB to YCbCr array for skin tone detection across skin types I-VI
    img_arr = np.array(img, dtype=np.float32)
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]

    # Standard YCbCr conversion formulas
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 128.0 + 0.500000 * r - 0.418688 * g - 0.081312 * b

    # Skin tone color range (Fitzpatrick types I to VI)
    # Cr: [133, 173], Cb: [77, 127], Y: [60, 255]
    skin_mask = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130) & (y >= 50)

    # Hair/scalp detection mask (dark/melanin or blonde/gray scalp hair textures)
    hair_mask = (y < 60) | ((r > 100) & (g > 80) & (b < 140))

    if image_type in (ImageType.scalp, ImageType.hair):
        relevant_pixels = np.sum(skin_mask | hair_mask)
    else:
        relevant_pixels = np.sum(skin_mask)

    total_pixels = img_arr.shape[0] * img_arr.shape[1]
    relevant_ratio = float(relevant_pixels / max(total_pixels, 1))

    if relevant_ratio < min_skin_hair_ratio:
        type_str = "Face" if image_type == ImageType.skin else "Scalp" if image_type in (ImageType.scalp, ImageType.hair) else "Body"
        return (
            False,
            f"No relevant {image_type.value} content detected in image (detected only {int(relevant_ratio * 100)}% skin/scalp presence). "
            f"Please upload only related clinical images ({type_str} photo for {image_type.value} analysis)."
        )

    return True, None
