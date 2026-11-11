import io

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., :3].astype(np.float32).mean(axis=2)
    return img


def window(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def standardize(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    return resize_max_side(window(to_gray(img)), max_side)


def is_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def read_dicom(data: bytes) -> np.ndarray:
    import pydicom

    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    ds = pydicom.dcmread(io.BytesIO(data))
    arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    if arr.ndim == 3 and int(ds.get("SamplesPerPixel", 1)) == 1:
        arr = arr[0]
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def decode(data: bytes) -> np.ndarray:
    if is_dicom(data):
        return read_dicom(data)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("unsupported image format")
    return arr


def read_image(path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"cannot read {path}")
    return arr
