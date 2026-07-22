import io
import numpy as np
from PIL import Image
from app.schemas import ImageType
from app.subject_validation import validate_subject_relevance


def create_skin_test_image() -> bytes:
    """Create a test image with realistic human skin tone RGB values."""
    # Typical Caucasian / Asian / South Asian skin tone (R > G > B, YCbCr in skin mask)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 0] = 210  # Red
    arr[:, :, 1] = 160  # Green
    arr[:, :, 2] = 130  # Blue
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_unrelated_object_image() -> bytes:
    """Create an unrelated non-clinical image (e.g. solid blue object / landscape)."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 0] = 10   # Red
    arr[:, :, 1] = 30   # Green
    arr[:, :, 2] = 220  # Blue
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_skin_image_passes_subject_validation():
    img_bytes = create_skin_test_image()
    is_valid, msg = validate_subject_relevance(img_bytes, ImageType.skin)
    assert is_valid is True
    assert msg is None


def test_unrelated_object_image_fails_subject_validation():
    img_bytes = create_unrelated_object_image()
    is_valid, msg = validate_subject_relevance(img_bytes, ImageType.skin)
    assert is_valid is False
    assert "No relevant skin content detected" in msg
    assert "Please upload only related clinical images" in msg
