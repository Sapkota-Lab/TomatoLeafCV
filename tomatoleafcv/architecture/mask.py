from pathlib import Path

import cv2
import numpy as np


def load_plant_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read mask: {path}")
    if not np.all((image == 0) | (image == 255)):
        raise ValueError("Expected a binary mask with a white plant on black")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    if count < 2:
        raise ValueError("The mask contains no white foreground")
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, width, height, area = stats[largest]
    if (
        x == 0 and y == 0 and width == image.shape[1] and height == image.shape[0]
        and area > image.size / 2
    ):
        raise ValueError(
            "The white foreground appears to be the background; "
            "invert the mask so the plant is white"
        )
    return np.uint8(labels == largest) * 255
