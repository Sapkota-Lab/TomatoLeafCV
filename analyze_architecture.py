import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


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


def leaflet_candidates(mask: np.ndarray) -> list[dict]:
    plant_area = cv2.countNonZero(mask)
    size = max(3, int(round(max(mask.shape) * 0.011)))
    size += 1 - size % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    minimum_area = max(size * size, int(plant_area * 0.0007))
    keep = stats[:, cv2.CC_STAT_AREA] >= minimum_area
    keep[0] = False
    seeds = np.uint8(keep[labels]) * 255
    if cv2.countNonZero(seeds) == 0:
        raise ValueError("No leaflet candidates found in the plant mask")

    distances, nearest = cv2.distanceTransformWithLabels(
        cv2.bitwise_not(seeds),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )
    candidates = []
    for label in np.unique(nearest[seeds != 0]):
        region = np.uint8(
            (nearest == label) & (mask != 0) & (distances <= size / 2 + 1)
        )
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue
        moments = cv2.moments(contour)
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        candidates.append(
            {
                "contour": contour,
                "center": center,
                "area_pixels": area,
                "perimeter_pixels": perimeter,
                "circularity": 4 * np.pi * area / (perimeter * perimeter),
            }
        )
    if not candidates:
        raise ValueError("No measurable leaflet candidates found")
    return sorted(candidates, key=lambda candidate: candidate["center"][1])


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


def analyze(mask: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    candidates = leaflet_candidates(mask)
    points, scale, axis, root_distance, sources = skeleton_graph(mask)
    tree = cKDTree(points)
    axis_nodes = set(axis.tolist())
    axis_points = points[axis]
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
                "junction": (int(junction_x / scale), int(junction_y / scale)),
                "leaflets": [],
            }
            groups.append(group)
        group["leaflets"].append(candidate)
        candidate["branch"] = group["id"]

    results = {
        "branch_count_estimate": len(groups),
        "leaflet_count_estimate": len(candidates),
        "overall_mean_circularity": float(
            np.mean([candidate["circularity"] for candidate in candidates])
        ),
        "branches": [
            {
                "id": group["id"],
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


def create_preview(mask: np.ndarray, candidates: list[dict], groups: list[dict]) -> np.ndarray:
    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    text_scale = max(0.4, max(mask.shape) / 2800)
    thickness = max(1, round(text_scale * 2))
    occupied = []
    for candidate in candidates:
        cv2.drawContours(preview, [candidate["contour"]], -1, (0, 0, 255), thickness)
    for group in groups:
        cv2.circle(preview, group["junction"], 3 * thickness, (0, 255, 0), -1)
    for candidate in candidates:
        x, y = (int(value) for value in candidate["center"])
        label = f"L{candidate['id']}"
        if candidate["branch"] is not None:
            label += f"/B{candidate['branch']}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness
        )
        label_x, label_y = x, y
        placed = False
        for distance in range(7):
            for dx, dy in ((0, -distance), (distance, 0), (-distance, 0), (0, distance)):
                tx = min(max(0, x + dx * (text_width + 4)), mask.shape[1] - text_width - 1)
                ty = min(max(text_height, y + dy * (text_height + baseline + 4)), mask.shape[0] - baseline - 1)
                box = (tx, ty - text_height, tx + text_width, ty + baseline)
                if all(
                    box[2] < other[0] or box[0] > other[2]
                    or box[3] < other[1] or box[1] > other[3]
                    for other in occupied
                ):
                    label_x, label_y = tx, ty
                    occupied.append(box)
                    placed = True
                    break
            if placed:
                break
        if label_x != x or label_y != y:
            cv2.line(preview, (x, y), (label_x, label_y), (0, 255, 255), 1)
        cv2.putText(
            preview, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            text_scale, (0, 0, 0), thickness + 3,
        )
        cv2.putText(
            preview, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            text_scale, (0, 255, 255), thickness,
        )
    return preview


def show_preview(preview: np.ndarray) -> None:
    cv2.namedWindow("Architecture estimates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Architecture estimates", 700, 900)
    cv2.imshow("Architecture estimates", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate leaflet and lateral branch architecture from a binary mask. "
            "Results are experimental and require visual inspection."
        ),
        epilog=(
            "The preview outlines candidate leaflets in red. L numbers identify "
            "leaflets, B numbers identify estimated branches, and green dots mark "
            "estimated branch junctions. Press any key to close the preview. "
            "Leaflet circularity is 4*pi*area/perimeter^2."
        ),
    )
    parser.add_argument("mask", type=Path, help="Binary PNG: white plant on black")
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Print results without opening a window",
    )
    args = parser.parse_args()
    try:
        mask = load_plant_mask(args.mask)
        results, candidates, groups = analyze(mask)
        print(json.dumps(results, indent=2))
        if not args.no_preview:
            show_preview(create_preview(mask, candidates, groups))
    except (ValueError, OSError, cv2.error) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    test_image = Path("test_overlay.png")
    test_results, test_candidates, test_groups = analyze(load_plant_mask(test_image))
    #Display the analyzed preview
    analyzed_preview = create_preview(load_plant_mask(test_image), test_candidates, test_groups)
    cv2.imshow("Analyzed Preview", analyzed_preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #Save the analyzed preview
    test_saved_preview = Path("test_architecture.png")
    cv2.imwrite(str(test_saved_preview), analyzed_preview)
