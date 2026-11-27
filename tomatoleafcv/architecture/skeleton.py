import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize


def skeleton_graph(mask: np.ndarray):
    height, width = mask.shape
    scale = min(1.0, 1000 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reduced = cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 127
    skeleton = skeletonize(reduced)
    points = np.argwhere(skeleton)
    if len(points) < 2:
        raise ValueError("The plant is too small to trace a stem")

    indices = np.full(skeleton.shape, -1, dtype=np.int32)
    indices[points[:, 0], points[:, 1]] = np.arange(len(points))
    rows = []
    columns = []
    weights = []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        neighbor_y = points[:, 0] + dy
        neighbor_x = points[:, 1] + dx
        valid = (
            (neighbor_y >= 0)
            & (neighbor_y < skeleton.shape[0])
            & (neighbor_x >= 0)
            & (neighbor_x < skeleton.shape[1])
        )
        source = np.flatnonzero(valid)
        target = indices[neighbor_y[valid], neighbor_x[valid]]
        connected = target >= 0
        source = source[connected]
        target = target[connected]
        rows.extend((source, target))
        columns.extend((target, source))
        weights.extend((np.full(len(source), np.hypot(dx, dy)),) * 2)

    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(points), len(points)),
    ).tocsr()
    root = int(np.argmax(points[:, 0]))
    tip = int(np.argmin(points[:, 0]))
    root_distance, previous = dijkstra(graph, indices=root, return_predecessors=True)
    if not np.isfinite(root_distance[tip]):
        raise ValueError(
            "The reduced skeleton is disconnected; the stem could not be traced"
        )
    axis = [tip]
    while axis[-1] != root:
        parent = int(previous[axis[-1]])
        if parent < 0:
            raise ValueError("Could not trace a path between the plant base and tip")
        axis.append(parent)
    axis = np.asarray(axis)
    _, _, sources = dijkstra(
        graph, indices=axis, min_only=True, return_predecessors=True
    )
    return points, scale, axis, root_distance, sources
