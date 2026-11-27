import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from tomatoleafcv.overlay import create_overlay
from tomatoleafcv.overlay_cli import IMAGE_EXTENSIONS, _is_within, _process_image, main
from tomatoleafcv.segmentation import (
    MAX_LEAF_HOLE_AREA,
    MIN_LEAF_COMPONENT_AREA,
    create_leaf_mask,
    create_leaf_mask_preview,
)


if __name__ == "__main__":
    test_image = Path(r"images\BGV004584\BGV004584-1-F0001.tif")
    image = cv2.imread(str(test_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(test_image)
    test_saved_image = Path(r"test_overlay.png")
    cv2.imwrite(str(test_saved_image), create_leaf_mask(image))
