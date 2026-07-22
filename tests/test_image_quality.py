import io
import numpy as np
from PIL import Image
from app.image_quality import validate_image_quality


def create_test_image(mode="normal") -> bytes:
    if mode == "normal":
        # Realistic image with gradient / features
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    elif mode == "solid":
        # Solid gray
        arr = np.full((224, 224, 3), (128, 128, 128), dtype=np.uint8)
        img = Image.fromarray(arr)
    elif mode == "dark":
        # Pitch dark with slight variation
        base = np.full((224, 224, 3), 5, dtype=np.uint8)
        noise = np.random.randint(0, 10, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    elif mode == "bright":
        # Overexposed white with slight variation
        base = np.full((224, 224, 3), 245, dtype=np.uint8)
        noise = np.random.randint(0, 10, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    elif mode == "blur":
        # Low contrast uniform blur
        arr = np.tile(np.linspace(100, 105, 224, dtype=np.uint8), (224, 1)).repeat(3).reshape(224, 224, 3)
        img = Image.fromarray(arr)
    else:
        raise ValueError(f"Unknown mode {mode}")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_valid_image_passes_quality_check():
    img_bytes = create_test_image("normal")
    is_valid, reason = validate_image_quality(img_bytes)
    assert is_valid is True
    assert reason is None


def test_solid_color_fails_quality_check():
    img_bytes = create_test_image("solid")
    is_valid, reason = validate_image_quality(img_bytes)
    assert is_valid is False
    assert "solid" in reason.lower() or "variance" in reason.lower()


def test_pitch_dark_image_fails_quality_check():
    img_bytes = create_test_image("dark")
    is_valid, reason = validate_image_quality(img_bytes)
    assert is_valid is False
    assert "dark" in reason.lower() or "solid" in reason.lower()


def test_overexposed_image_fails_quality_check():
    img_bytes = create_test_image("bright")
    is_valid, reason = validate_image_quality(img_bytes)
    assert is_valid is False
    assert "overexposed" in reason.lower() or "solid" in reason.lower()
