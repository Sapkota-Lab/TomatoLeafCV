import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MIN_LEAF_COMPONENT_AREA = 500
MAX_LEAF_HOLE_AREA = 100


def create_leaf_mask(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a color image with three BGR channels")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_pixels = cv2.inRange(hsv_image, (25, 35, 40), (100, 255, 255))
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


def create_overlay(image: np.ndarray) -> np.ndarray:
    mask = create_leaf_mask(image)
    highlight = np.full_like(image, (0, 0, 255))
    blended = cv2.addWeighted(image, 0.55, highlight, 0.45, 0)

    overlay = image.copy()
    leaf_pixels = mask != 0
    overlay[leaf_pixels] = blended[leaf_pixels]
    return overlay


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _process_image(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError("OpenCV could not read this image")

    overlay = create_overlay(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError("OpenCV could not write the overlay")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Highlight green pixels (HSV hue 25-100) in an image or directory."
        ),
        epilog=(
            "Examples:\n"
            "  python create_overlay.py images\\leaf.tif\n"
            "  python create_overlay.py images -o overlays"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="An image file or a directory containing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output PNG file for one image, or output directory for a directory "
            "(default: alongside input)"
        ),
    )
    args = parser.parse_args(argv)
    input_path = args.input

    if not input_path.exists():
        parser.error("input path does not exist: {}".format(input_path))

    if input_path.is_dir():
        output_directory = (
            args.output
            if args.output is not None
            else input_path.with_name(input_path.name + "_overlays")
        )
        resolved_output = output_directory.resolve()
        image_paths = [
            path
            for path in sorted(input_path.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not _is_within(path.resolve(), resolved_output)
        ]
        if not image_paths:
            parser.error("no supported image files found in: {}".format(input_path))

        output_paths = [
            output_directory / path.relative_to(input_path).with_name(
                "{}_{}_overlay.png".format(path.stem, path.suffix[1:].lower())
            )
            for path in image_paths
        ]
    else:
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            parser.error("unsupported image file type: {}".format(input_path))
        output_path = args.output or input_path.with_name(
            input_path.stem + "_overlay.png"
        )
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / (input_path.stem + "_overlay.png")
        image_paths = [input_path]
        output_paths = [output_path]

    failures = 0
    for image_path, output_path in zip(image_paths, output_paths):
        try:
            _process_image(image_path, output_path)
        except (OSError, ValueError, cv2.error) as error:
            print(
                "Could not process {}: {}".format(image_path, error),
                file=sys.stderr,
            )
            failures += 1
        else:
            print("Created {}".format(output_path))

    return 1 if failures else 0


if __name__ == "__main__":
    test_image = Path(r"images\BGV004584\BGV004584-1-F0001.tif")
    image = cv2.imread(str(test_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(test_image)
    test_saved_image = Path(r"images\BGV004584\BGV004584-1-F0001_overlay.png")
    cv2.imwrite(str(test_saved_image), create_leaf_mask(image))