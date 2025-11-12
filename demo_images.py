"""
demo_images.py

Utilities to load demo images for the Road Hazard Detection backend.

Features:
- Uses pathlib for cross-platform paths (no invalid escape sequence warnings).
- Returns images as base64-encoded JPEG strings.
- Supports:
    * a single file path
    * a directory (loads all image files in it)
    * a built-in default list (modifiable)
- Robust error handling and optional logging.
"""

from __future__ import annotations
import base64
from pathlib import Path
from typing import List, Dict, Optional
import logging
import cv2
import numpy as np

# Setup simple logger (module-level)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default demo images for testing (can be overridden by user input).
# Use forward slashes or Path to avoid escape-sequence warnings on Windows.
DEFAULT_DEMO_IMAGES = [
    {"name": "pothole_demo.jpg",         "path": Path("D:/volks1/images/patho.png"),  "description": "Road with potholes"},
    {"name": "stalled_vehicle_demo.jpg", "path": Path("D:/volks1/images/sta.png"),    "description": "Stalled vehicle with license plate"},
    {"name": "speed_breaker_demo.jpg",   "path": Path("D:/volks1/images/ali.png"),    "description": "Road with speed breaker"},
    {"name": "manhole_demo.jpg",         "path": Path("D:/volks1/images/manhole.png"), "description": "Road with manhole"},
    {"name": "Aligator_demo.jpg",        "path": Path("D:/volks1/images/ali.png"),    "description": "Road with debris"},
    {"name": "face_blur_demo.jpg",       "path": Path("D:/volks1/images/man.png"),    "description": "Image with faces to blur"},
    {"name": "plate_blur_demo.jpg",      "path": Path("D:/volks1/images/plate.png"),  "description": "Image with license plates to blur"},
]

# Valid image extensions (lowercase)
_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _encode_image_to_base64(img: np.ndarray) -> Optional[str]:
    """
    Encode a BGR OpenCV image (numpy array) to a base64 JPEG string.
    Returns None on failure.
    """
    try:
        success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            logger.warning("cv2.imencode failed for image")
            return None
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.exception("Failed to encode image to base64: %s", e)
        return None


def get_demo_images(demo_path: Optional[str | Path] = None) -> List[Dict]:
    """
    Return a list of demo images with base64 encoded data.

    Args:
        demo_path: Optional path to a file or directory. If None, returns default list.

    Returns:
        List of dicts: { "name": str, "data": str (base64) or "", "description": str, "path": str }
    """
    if demo_path:
        return get_demo_images_from_path(demo_path)
    return get_demo_images_from_list(DEFAULT_DEMO_IMAGES)


def get_demo_images_from_path(demo_path: str | Path) -> List[Dict]:
    """
    Load demo images from a specified path (file or directory).
    If no valid images are found, falls back to defaults.
    """
    p = Path(demo_path)
    images: List[Dict] = []

    if p.is_file():
        logger.info("Loading single demo image: %s", p)
        try:
            img = cv2.imread(str(p))
            if img is not None:
                b64 = _encode_image_to_base64(img)
                images.append({"name": p.name, "data": b64 or "", "description": f"Demo image: {p.name}", "path": str(p)})
            else:
                logger.warning("cv2.imread returned None for file: %s", p)
        except Exception:
            logger.exception("Failed loading image file: %s", p)

    elif p.is_dir():
        logger.info("Loading demo images from directory: %s", p)
        for child in sorted(p.iterdir()):
            if child.suffix.lower() in _VALID_EXTS and child.is_file():
                try:
                    img = cv2.imread(str(child))
                    if img is not None:
                        b64 = _encode_image_to_base64(img)
                        images.append({"name": child.name, "data": b64 or "", "description": f"Demo image: {child.name}", "path": str(child)})
                    else:
                        logger.warning("Could not read image (cv2 returned None): %s", child)
                        images.append({"name": child.name, "data": "", "description": f"Unreadable image: {child.name}", "path": str(child)})
                except Exception:
                    logger.exception("Error loading image: %s", child)
                    images.append({"name": child.name, "data": "", "description": f"Error loading: {child.name}", "path": str(child)})
    else:
        logger.warning("Path does not exist: %s", p)

    if not images:
        # Fall back to defaults
        logger.info("No valid images found at %s — falling back to default demo images", p)
        return get_demo_images_from_list(DEFAULT_DEMO_IMAGES)

    logger.info("Loaded %d demo images from %s", len(images), p)
    return images


def get_demo_images_from_list(image_list: List[Dict]) -> List[Dict]:
    """
    Return list of demo images from a predefined list.
    Each item in image_list should be a dict with keys: name, path (Path or str), description.
    Returns items with 'data' set to base64 string or empty string if file missing/unreadable.
    """
    images: List[Dict] = []
    for entry in image_list:
        name = entry.get("name", "unknown")
        raw_path = entry.get("path")
        description = entry.get("description", "")
        try:
            path_obj = Path(raw_path) if not isinstance(raw_path, Path) else raw_path
            if path_obj.exists() and path_obj.is_file() and path_obj.suffix.lower() in _VALID_EXTS:
                img = cv2.imread(str(path_obj))
                if img is not None:
                    b64 = _encode_image_to_base64(img)
                    images.append({"name": name, "data": b64 or "", "description": description, "path": str(path_obj)})
                else:
                    logger.warning("cv2.imread returned None for default image: %s", path_obj)
                    images.append({"name": name, "data": "", "description": description, "path": str(path_obj)})
            else:
                logger.debug("Default image path missing or invalid: %s", path_obj)
                images.append({"name": name, "data": "", "description": description, "path": str(path_obj)})
        except Exception:
            logger.exception("Error processing default image entry: %s", entry)
            images.append({"name": name, "data": "", "description": description, "path": str(raw_path)})

    return images


# Backwards-compatible alias
def get_demo_images_default() -> List[Dict]:
    """Deprecated alias — use get_demo_images() instead."""
    logger.warning("get_demo_images_default() is deprecated; use get_demo_images()")
    return get_demo_images()
