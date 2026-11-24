#!/usr/bin/env python3
"""Render an LDraw model as a falling build-step animation."""

import argparse
from pathlib import Path

from bricknet_render.data import default_glb_dir
from bricknet_render.render import TurntableOptions, render_ldr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="LDraw model",
    )
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument(
        "--glb-dir",
        type=Path,
        default=default_glb_dir(),
        help="Directory of pre-built .glb parts (default: <BrickNet data dir>/glb)",
    )
    parser.add_argument("--format", default="png8", choices=["png8", "png16", "mp4", "mov", "gif"])
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--rotation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rotation-period", type=float, default=4.0)
    parser.add_argument("--turn-direction", choices=["cw", "ccw"], default="cw")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--resolution", default="512x512")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--temporal-denoise", action="store_true")
    parser.add_argument("--ffmpeg", help="FFmpeg executable for GIF encoding")
    parser.add_argument("--save-blend", type=Path, help="Optional path for the generated Blender scene")
    args = parser.parse_args()

    outputs = render_ldr(
        args.input,
        args.output,
        glb_dir=args.glb_dir,
        samples=args.samples,
        resolution=args.resolution,
        device=args.device,
        save_blend=None if args.save_blend is None else str(args.save_blend),
        turntable=TurntableOptions(
            format=args.format,
            fps=args.fps,
            duration=args.duration,
            rotation_period=args.rotation_period,
            clockwise=args.turn_direction == "cw",
            rotate=args.rotation,
            build_steps=True,
            falling=True,
            temporal_denoise=args.temporal_denoise,
            ffmpeg=args.ffmpeg,
        ),
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
