#!/usr/bin/env python3
"""Render a labeled mosaic of catalog materials on one LDraw part."""

import argparse
import math
import tempfile
import time
from pathlib import Path

import bpy

# bpy initializes Blender's built-in font and image-buffer modules.
import blf
import imbuf
import numpy as np

from bricknet_render.data import default_glb_dir
from bricknet_render.material_catalog import material_specs
from bricknet_render.render import RenderOptions, build_scene, configure_image_format, get_or_build_material, save_blend


def save_catalog_blend(options: RenderOptions, materials: list[bpy.types.Material]) -> None:
    """Retain every catalog material in the saved .blend."""

    if options.save_blend is None:
        return
    fake_user_states = [(material, material.use_fake_user) for material in materials]
    try:
        for material, _state in fake_user_states:
            material.use_fake_user = True
        save_blend(options)
    finally:
        for material, state in fake_user_states:
            material.use_fake_user = state


def render_materials(args: argparse.Namespace, render_dir: Path) -> dict[str, Path]:
    """Render each catalog material on one part."""

    with tempfile.TemporaryDirectory(prefix="bricknet-materials-") as temp_dir:
        stub = Path(temp_dir) / "part.ldr"
        stub.write_text("1 4 0 0 0 1 0 0 0 1 0 0 0 1 %s.dat\n" % args.part, encoding="utf-8")
        options = RenderOptions(
            ldr_path=str(stub),
            out_dir=str(render_dir),
            glb_dir=str(args.glb_dir),
            format="png",
            samples=args.samples,
            resolution=args.resolution,
            views=1,
            device=args.device,
            save_blend=None if args.save_blend is None else str(args.save_blend),
        )
        mesh_objs, _points, _yaws = build_scene(options)

    configure_image_format(options)
    scene = bpy.context.scene
    scene.frame_set(0)
    scene.render.image_settings.file_format = "PNG"

    outputs: dict[str, Path] = {}
    materials: list[bpy.types.Material] = []
    start = time.time()
    for name in sorted(material_specs()):
        material = get_or_build_material(name)
        for obj in mesh_objs:
            obj.data.materials.clear()
            obj.data.materials.append(material)
        out_path = render_dir / ("%s.png" % name)
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        outputs[name] = out_path
        materials.append(material)
    elapsed = time.time() - start
    print("%d materials in %.1fs (%.2fs each)" % (len(outputs), elapsed, elapsed / len(outputs)))
    save_catalog_blend(options, materials)
    return outputs


def load_image_pixels(path: Path) -> tuple[tuple[int, int], np.ndarray]:
    """Load one image as a bottom-up Blender RGBA array."""

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        return tuple(image.size), pixels.reshape(image.size[1], image.size[0], 4)
    finally:
        bpy.data.images.remove(image)


def build_mosaic(outputs: dict[str, Path], out_path: Path, label_height: int = 18) -> None:
    """Tile per-material renders into one labeled image."""

    names = sorted(outputs)
    (width, height), first_pixels = load_image_pixels(outputs[names[0]])
    columns = math.ceil(math.sqrt(len(names)))
    rows = math.ceil(len(names) / columns)
    cell_height = height + label_height
    mosaic_pixels = np.ones((rows * cell_height, columns * width, 4), dtype=np.float32)
    labels = []
    for index, name in enumerate(names):
        x = (index % columns) * width
        y = (rows - 1 - index // columns) * cell_height
        pixels = first_pixels if index == 0 else load_image_pixels(outputs[name])[1]
        mosaic_pixels[y : y + height, x : x + width] = pixels
        labels.append((name, x + 2, y + height + 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = bpy.data.images.new("Material Mosaic", columns * width, rows * cell_height, alpha=True)
    try:
        image.pixels.foreach_set(mosaic_pixels.ravel())
        image.filepath_raw = str(out_path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)

    mosaic = imbuf.load(str(out_path))
    try:
        with blf.bind_imbuf(0, mosaic):
            blf.size(0, 11)
            blf.color(0, 0.0, 0.0, 0.0, 1.0)
            for name, x, y in labels:
                blf.position(0, x, y, 0)
                blf.draw_buffer(0, name)

        imbuf.write(mosaic, filepath=str(out_path))
    finally:
        mosaic.free()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--glb-dir",
        type=Path,
        default=default_glb_dir(),
        help="Directory of pre-built .glb parts (default: <BrickNet data dir>/glb)",
    )
    parser.add_argument("--part", default="3001", help="LDraw part stem to render")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", default="240x160")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--save-blend", type=Path, help="Optional path for a scene containing every catalog material")
    args = parser.parse_args()

    render_dir = args.out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    outputs = render_materials(args, render_dir)
    mosaic_path = args.out_dir / "mosaic.png"
    build_mosaic(outputs, mosaic_path)
    print(mosaic_path)


if __name__ == "__main__":
    main()
