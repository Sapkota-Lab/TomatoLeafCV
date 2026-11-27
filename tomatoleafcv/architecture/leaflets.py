import cv2
import numpy as np


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
