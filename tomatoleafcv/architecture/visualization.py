import cv2
import numpy as np

from .skeleton import skeleton_graph


def create_architecture_drawing(mask: np.ndarray, groups: list[dict]) -> np.ndarray:
    points, scale, axis, _, _ = skeleton_graph(mask)
    height, width = mask.shape
    reduced = np.zeros(
        (max(1, round(height * scale)), max(1, round(width * scale)), 3),
        dtype=np.uint8,
    )
    reduced[points[:, 0], points[:, 1]] = (255, 255, 255)
    drawing = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
    axis_xy = np.rint(points[axis, ::-1] / scale).astype(np.int32)
    cv2.polylines(drawing, [axis_xy], False, (0, 0, 255), max(1, round(2 / scale)))
    text_scale = max(0.4, max(mask.shape) / 2800)
    thickness = max(1, round(text_scale * 2))
    for group in groups:
        x, y = group["junction"]
        cv2.circle(drawing, (x, y), 3 * thickness, (0, 255, 0), -1)
        cv2.putText(
            drawing, f"B{group['id']}", (x + 4 * thickness, y),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), thickness,
        )
    return drawing


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
