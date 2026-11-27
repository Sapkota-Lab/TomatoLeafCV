import cv2
import numpy as np

from .segmentation import create_leaf_mask


def create_overlay(image: np.ndarray) -> np.ndarray:
    mask = create_leaf_mask(image)
    highlight = np.full_like(image, (0, 0, 255))
    blended = cv2.addWeighted(image, 0.55, highlight, 0.45, 0)

    overlay = image.copy()
    leaf_pixels = mask != 0
    overlay[leaf_pixels] = blended[leaf_pixels]
    return overlay
