import cv2
import numpy as np

MIN_LEAF_COMPONENT_AREA = 500
MAX_LEAF_HOLE_AREA = 100


def create_leaf_mask(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a color image with three BGR channels")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_pixels = cv2.inRange(hsv_image, (25, 60, 40), (100, 255, 255))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        green_pixels, connectivity=8
    )
    retained_components = np.zeros(component_count, dtype=np.uint8)
    retained_components[1:] = (
        stats[1:, cv2.CC_STAT_AREA] >= MIN_LEAF_COMPONENT_AREA
    ).astype(np.uint8) * 255
    mask = retained_components[labels]

    _, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(mask), connectivity=8
    )
    small_holes = hole_stats[:, cv2.CC_STAT_AREA] <= MAX_LEAF_HOLE_AREA
    small_holes[0] = False
    small_holes[hole_labels[0, :]] = False
    small_holes[hole_labels[-1, :]] = False
    small_holes[hole_labels[:, 0]] = False
    small_holes[hole_labels[:, -1]] = False
    mask[small_holes[hole_labels]] = 255
    return mask


def create_leaf_mask_preview(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(create_leaf_mask(image))
