import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Create leaf overlays or analyze plant architecture.",
        epilog="Run 'python main.py COMMAND --help' for command-specific options.",
    )
    parser.add_argument("command", choices=("overlay", "architecture"))
    args = parser.parse_args(argv[:1])

    if args.command == "overlay":
        from tomatoleafcv.overlay_cli import main as overlay_main

        return overlay_main(argv[1:], prog=f"{parser.prog} overlay")

    from tomatoleafcv.architecture.cli import main as architecture_main

    return architecture_main(argv[1:], prog=f"{parser.prog} architecture")


if __name__ == "__main__":
    sys.exit(main())
