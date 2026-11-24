"""Render LDraw models or fetch the part meshes."""

import argparse
import sys
from pathlib import Path

from . import __version__
from .data import default_glb_dir, fetch_glbs


def main(argv: list[str] | None = None) -> None:
    """Render a model, or install part meshes with ``fetch-glbs``."""

    argv = sys.argv[1:] if argv is None else argv
    prog = "bricknet-render" if Path(sys.argv[0]).stem == "bricknet-render" else "python -m bricknet_render"
    if argv[:1] != ["fetch-glbs"]:
        from .render import main as render_main

        render_main(argv, prog=prog)
        return

    parser = argparse.ArgumentParser(prog=prog + " fetch-glbs")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--dest", type=Path, help="extract here instead of <BrickNet data dir>/glb")
    args = parser.parse_args(argv[1:])

    destination = default_glb_dir() if args.dest is None else args.dest
    print("downloading GLB part meshes (323 MiB) -> %s" % destination)
    fetch_glbs(destination)
    print("installed GLB part meshes")


if __name__ == "__main__":
    main()
