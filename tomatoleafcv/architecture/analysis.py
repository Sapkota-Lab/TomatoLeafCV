from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .leaflets import leaflet_candidates
from .skeleton import skeleton_graph


def analyze(
    mask: np.ndarray, color_image: Optional[np.ndarray] = None
) -> tuple[dict, list[dict], list[dict]]:
    if color_image is not None and (
        color_image.ndim != 3 or color_image.shape[2] != 3
        or color_image.shape[:2] != mask.shape
    ):
        raise ValueError("The color image must have three channels and match the mask size")

    candidates = leaflet_candidates(mask)
    points, scale, axis, root_distance, sources = skeleton_graph(mask)
    tree = cKDTree(points)
    axis_nodes = set(axis.tolist())
    axis_points = points[axis]
    base_to_tip = axis[::-1]
    steps = np.linalg.norm(np.diff(points[base_to_tip], axis=0), axis=1) / scale
    axis_positions = dict(zip(base_to_tip, np.r_[0.0, np.cumsum(steps)]))
    groups = []
    axis_tolerance = max(3.0, (axis_points[:, 0].max() - axis_points[:, 0].min()) * 0.02)

    for number, candidate in enumerate(candidates, start=1):
        x, y = candidate["center"]
        _, node = tree.query((y * scale, x * scale))
        node = int(node)
        source = int(sources[node])
        if source < 0:
            raise ValueError("A leaflet candidate is disconnected from the traced stem")
        junction_y, junction_x = points[source]
        candidate["id"] = number
        candidate["branch"] = None
        if node in axis_nodes:
            continue

        side = "left" if x * scale < junction_x else "right"
        position = float(root_distance[source])
        group = next(
            (
                group
                for group in groups
                if group["side"] == side
                and abs(group["position"] - position) <= axis_tolerance
            ),
            None,
        )
        if group is None:
            group = {
                "id": len(groups) + 1,
                "side": side,
                "position": position,
                "axis_position": float(axis_positions[source]),
                "junction": (int(junction_x / scale), int(junction_y / scale)),
                "leaflets": [],
            }
            groups.append(group)
        group["leaflets"].append(candidate)
        candidate["branch"] = group["id"]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        raise ValueError("The plant outline has no measurable area or perimeter")

    ordered_groups = sorted(groups, key=lambda group: group["axis_position"])
    branch_spacings = [
        {
            "from_branch_id": first["id"],
            "to_branch_id": second["id"],
            "distance_pixels": second["axis_position"] - first["axis_position"],
        }
        for first, second in zip(ordered_groups, ordered_groups[1:])
    ]
    main_axis_color = None
    if color_image is not None:
        coords = np.rint(axis_points / scale).astype(int)
        coords[:, 0] = np.clip(coords[:, 0], 0, mask.shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, mask.shape[1] - 1)
        coords = coords[mask[coords[:, 0], coords[:, 1]] != 0]
        if len(coords) == 0:
            raise ValueError("No foreground pixels found along the traced main axis")
        mean_bgr = np.mean(color_image[coords[:, 0], coords[:, 1]], axis=0)
        main_axis_color = {
            "mean_rgb": [float(value) for value in mean_bgr[::-1]],
            "sample_count": len(coords),
        }

    results = {
        "branch_count_estimate": len(groups),
        "leaflet_count_estimate": len(candidates),
        "plant_outline_circularity": float(4 * np.pi * area / (perimeter * perimeter)),
        "main_axis_length_pixels": float(sum(steps)),
        "main_axis_color": main_axis_color,
        "consecutive_branch_spacings": branch_spacings,
        "overall_mean_circularity": float(
            np.mean([candidate["circularity"] for candidate in candidates])
        ),
        "branches": [
            {
                "id": group["id"],
                "side": group["side"],
                "junction_xy_pixels": group["junction"],
                "distance_from_base_pixels": group["axis_position"],
                "leaflet_ids": [leaflet["id"] for leaflet in group["leaflets"]],
                "mean_circularity": float(
                    np.mean([leaflet["circularity"] for leaflet in group["leaflets"]])
                ),
            }
            for group in groups
        ],
        "leaflets": [
            {
                "id": candidate["id"],
                "branch_id": candidate["branch"],
                "area_pixels": candidate["area_pixels"],
                "perimeter_pixels": candidate["perimeter_pixels"],
                "circularity": candidate["circularity"],
            }
            for candidate in candidates
        ],
    }
    return results, candidates, groups
