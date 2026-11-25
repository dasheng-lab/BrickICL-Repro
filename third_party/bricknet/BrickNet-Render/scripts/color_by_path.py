#!/usr/bin/env python3
"""Color parts by BrickNet breadth-first traversal order."""

import argparse
from dataclasses import replace
from pathlib import Path

import bpy
import matplotlib
from bricknet import Graph, parse_ldr, sample_tree
from matplotlib.colors import to_hex

from bricknet_render.data import default_glb_dir
from bricknet_render.render import (
    PART_SOURCE_INDEX_PROP,
    MaterialMapper,
    MaterialVariant,
    RenderOptions,
    build_scene,
    configure_image_format,
    render_ldr,
    save_blend,
)


def bricknet_bfs_indices(graph: Graph) -> tuple[int, ...]:
    """Return source part indices in BrickNet's BFS order, component by component."""

    # sample_tree returns Part values rather than source indices. Colors do not
    # affect traversal, so an indexed copy carries source positions through.
    indexed = replace(graph, colors=tuple(range(len(graph.part_ids))))
    return tuple(
        int(part.color)
        for component in range(len(indexed.components))
        for part in sample_tree(indexed, component, method="bfs").parts
    )


def traversal_colors(order: tuple[int, ...], cmap_name: str) -> dict[int, str]:
    """Map each source part index to an sRGB ``#RRGGBB`` colormap value."""

    cmap = matplotlib.colormaps[cmap_name]
    denominator = max(1, len(order) - 1)
    return {part_index: to_hex(cmap(rank / denominator)) for rank, part_index in enumerate(order)}


def animate_reveal(objects: list[bpy.types.Object], reveal_frame: dict[int, int]) -> None:
    """Hide every part, then reveal each on its own frame."""

    for obj in objects:
        for prop in ("hide_render", "hide_viewport"):
            setattr(obj, prop, True)
            obj.keyframe_insert(data_path=prop, frame=0)
            setattr(obj, prop, False)
            obj.keyframe_insert(data_path=prop, frame=reveal_frame[int(obj[PART_SOURCE_INDEX_PROP])])


def render_graph_walk(options: RenderOptions, order: tuple[int, ...], mapper: MaterialMapper) -> list[Path]:
    """Frame the complete model, then reveal one graph-walk placement per frame."""

    objects, _points, _yaws = build_scene(options, map_material=mapper)
    animate_reveal(
        [obj for obj in objects if not obj.hide_render],
        {source_index: frame for frame, source_index in enumerate(order, 1)},
    )

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, len(order)
    scene.frame_set(0)
    output = Path(options.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    configure_image_format(options)
    scene.render.filepath = str(output / (Path(options.ldr_path).stem + "_"))
    save_blend(options)
    outputs = [Path(scene.render.frame_path(frame=frame)) for frame in range(scene.frame_end + 1)]
    bpy.ops.render.render(animation=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="LDraw model")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--glb-dir", type=Path, default=default_glb_dir(), help="Directory containing pre-built part GLBs"
    )
    parser.add_argument("--cmap", default="turbo", help="matplotlib colormap name")
    parser.add_argument("--yaw0", type=float, default=5.0, help="First view yaw in degrees")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--views", type=int, default=8)
    mode.add_argument("--build", action="store_true", help="Render an empty frame, then one part per graph-walk frame")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--resolution", default="512x512")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--save-blend", type=Path, help="Optional path for the generated Blender scene")
    args = parser.parse_args()

    graph = parse_ldr(args.input.read_text(encoding="utf-8"))
    order = bricknet_bfs_indices(graph)
    colors = traversal_colors(order, args.cmap)

    def mapper(slot):
        return MaterialVariant(colors[slot.placement.source_index])

    settings = dict(
        glb_dir=str(args.glb_dir),
        format="png",
        yaw0=args.yaw0,
        samples=args.samples,
        resolution=args.resolution,
        device=args.device,
        save_blend=None if args.save_blend is None else str(args.save_blend),
    )
    outputs = (
        render_graph_walk(
            RenderOptions(ldr_path=str(args.input), out_dir=str(args.out_dir), **settings),
            order,
            mapper,
        )
        if args.build
        else render_ldr(args.input, args.out_dir, views=args.views, map_material=mapper, **settings)
    )
    print(*outputs, sep="\n")


if __name__ == "__main__":
    main()
