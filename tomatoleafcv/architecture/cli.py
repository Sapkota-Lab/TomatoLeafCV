import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2

from .analysis import analyze
from .mask import load_plant_mask
from .visualization import create_architecture_drawing, create_preview, show_preview


def main(argv: Optional[list[str]] = None, *, prog: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Estimate leaflet and lateral branch architecture from a binary mask. "
            "Results are experimental and require visual inspection."
        ),
        epilog=(
            "The preview outlines candidate leaflets in red. L numbers identify "
            "leaflets, B numbers identify estimated branches, and green dots mark "
            "estimated branch junctions. Press any key to close the preview. "
            "Circularity is 4*pi*area/perimeter^2. Measurements are in pixels."
        ),
    )
    parser.add_argument(
        "mask", type=Path, nargs="?", default=Path("test_overlay.png"),
        help="Binary PNG: white plant on black (default: test_overlay.png)",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Print results without opening a window",
    )
    parser.add_argument(
        "--image", type=Path,
        help="Original color image aligned with the mask, for main-axis color",
    )
    parser.add_argument(
        "--drawing", type=Path,
        help="Save a skeleton drawing with the main axis and branch junctions",
    )
    legacy_mode = argv is None and len(sys.argv) == 1
    args = parser.parse_args(argv)
    try:
        mask = load_plant_mask(args.mask)
        image = None
        if args.image is not None:
            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read color image: {args.image}")
        results, candidates, groups = analyze(mask, image)
        if args.drawing is not None:
            if args.drawing.resolve() in {
                path.resolve() for path in (args.mask, args.image) if path is not None
            }:
                raise ValueError("The drawing path must not overwrite an input image")
            if not cv2.imwrite(str(args.drawing), create_architecture_drawing(mask, groups)):
                raise OSError(f"Could not write architecture drawing: {args.drawing}")
        if not args.no_preview:
            preview = create_preview(mask, candidates, groups)
            show_preview(preview)
            if legacy_mode:
                if not cv2.imwrite("test_architecture.png", preview):
                    raise OSError("Could not write test_architecture.png")
                with Path("test_architecture_results.json").open("w") as output:
                    json.dump(results, output, indent=2)
        print(json.dumps(results, indent=2))
    except (ValueError, OSError, cv2.error) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    return 0
