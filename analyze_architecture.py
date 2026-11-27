import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from tomatoleafcv.architecture.analysis import analyze
from tomatoleafcv.architecture.cli import main
from tomatoleafcv.architecture.leaflets import leaflet_candidates
from tomatoleafcv.architecture.mask import load_plant_mask
from tomatoleafcv.architecture.skeleton import skeleton_graph
from tomatoleafcv.architecture.visualization import (
    create_architecture_drawing,
    create_preview,
    show_preview,
)


if __name__ == "__main__":
    sys.exit(main())
