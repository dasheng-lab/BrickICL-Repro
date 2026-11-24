#!/usr/bin/env python3
"""Render a catalog of LDraw parts: one multi-view mosaic per part."""

import argparse
import math
import tempfile
import time
from pathlib import Path

import bpy
import numpy as np

from bricknet_render.data import default_glb_dir
from bricknet_render.render import RenderOptions, build_scene, configure_output, render_views


def part_names(args: argparse.Namespace, glb_dir: Path, out_dir: Path) -> list[str]:
    """Return this shard's parts, skipping those already rendered."""

    if args.parts:
        names = [line.strip() for line in args.parts.read_text(encoding="utf-8").splitlines() if line.strip()]
        names = [name.removesuffix(".dat") for name in names]
    else:
        names = sorted(path.stem for path in glb_dir.glob("*.glb"))
    names = names[args.shard :: args.shards]
    return [
        name for name in names if (glb_dir / ("%s.glb" % name)).exists() and not (out_dir / ("%s.png" % name)).exists()
    ]


def part_options(args: argparse.Namespace, ldr_path: Path, out_dir: Path) -> RenderOptions:
    """Return the catalog look: neutral studio lighting on a plain background."""

    return RenderOptions(
        ldr_path=str(ldr_path),
        out_dir=str(out_dir),
        glb_dir=str(args.glb_dir),
        format="png8",
        samples=args.samples,
        resolution=args.resolution,
        views=args.views,
        device=args.device,
        compute=args.compute,
        random_seed=0,
        light_preset="BUILD_FL",
        light_intensity=0.6,
        lighting_follows_camera=True,
        yaw0=35.0,
        pitch=-25.0,
        margin=1.05,
        field_of_view=22.0,
        env_map=None,
        env_color=(0.931, 1.0, 1.055),
        env_intensity=0.6,
        env_rotation=(0.0, 0.0, 20.0),
        env_effect=0.1,
    )


def load_image_pixels(path: Path) -> tuple[tuple[int, int], np.ndarray]:
    """Load one image as a bottom-up Blender RGBA array."""

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return tuple(image.size), pixels.reshape(image.size[1], image.size[0], 4)
    finally:
        bpy.data.images.remove(image)


def build_mosaic(frames: list[Path], out_path: Path) -> None:
    """Tile one part's views into a single image."""

    (width, height), first_pixels = load_image_pixels(frames[0])
    columns = max(len(frames) // 2, 1)
    rows = math.ceil(len(frames) / columns)
    mosaic_pixels = np.zeros((rows * height, columns * width, 4), dtype=np.float32)
    for index, frame in enumerate(frames):
        x = (index % columns) * width
        y = (rows - 1 - index // columns) * height
        mosaic_pixels[y : y + height, x : x + width] = first_pixels if index == 0 else load_image_pixels(frame)[1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = bpy.data.images.new("Part Mosaic", columns * width, rows * height, alpha=True)
    try:
        image.pixels.foreach_set(mosaic_pixels.ravel())
        image.filepath_raw = str(out_path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def render_parts(args: argparse.Namespace, out_dir: Path) -> None:
    """Render every part in this shard."""

    names = part_names(args, args.glb_dir, out_dir)
    if not names:
        return

    rendered = failed = 0
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="bricknet-parts-") as temp_dir:
        work = Path(temp_dir)
        for part in names:
            ldr_path = work / ("%s.ldr" % part)
            ldr_path.write_text(
                "0 %s\n0 Name: %s.ldr\n1 %d 0 0 0 1 0 0 0 1 0 0 0 1 %s.dat\n" % (part, part, args.color, part),
                encoding="utf-8",
            )
            for stale in work.glob("*.png"):
                stale.unlink()

            options = part_options(args, ldr_path, work)
            build_scene(options)
            configure_output(options)
            frames = [Path(path) for path in render_views(options) if Path(path).exists()]
            if len(frames) < args.views:
                print("FAILED %s (%d/%d frames)" % (part, len(frames), args.views), flush=True)
                failed += 1
                continue

            build_mosaic(frames, out_dir / ("%s.png" % part))
            rendered += 1

    elapsed = time.time() - start
    print("%d parts in %.1fs (%.2fs each)" % (rendered, elapsed, elapsed / max(rendered, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--glb-dir",
        type=Path,
        default=default_glb_dir(),
        help="Directory of pre-built .glb parts (default: <BrickNet data dir>/glb)",
    )
    parser.add_argument("--parts", type=Path, help="File of part names to render (default: every .glb)")
    parser.add_argument("--color", type=int, default=14, help="LDraw color code (14 = yellow)")
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--resolution", default="512x512")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--compute", default="AUTO")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1, help="Render a disjoint slice of the parts")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_parts(args, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
