"""Render side-by-side comparison figures for the cases listed in cases.json.

For each case, five panels share one camera:

  context     the input G \\ M
  reference   the true model
  frozen / plain / masked    G \\ M + what each model wrote (its best sample of 8, unless cases.json says otherwise)

Two styles are rendered:

  natural     every part in its own colour
  highlight   the context in light grey; added parts coloured by verdict
              (reference M blue; placed right green; right part, wrong place orange; not a part of M red)

Outputs, all next to this file:

  ldr/<case>/<style>/<n>_<panel>.ldr    the LDraw models, ready to open in BrickLink Studio
  panels/<case>/<style>_<panel>.png     single renders (BrickNet-Render, Blender Cycles)
  compare/<case>.png                    the panels of one case in a labelled grid
  compare/all_<style>.png               every case stacked, one row per case

Usage (repo root, bricknet env with bricknet-render):
  python nzd-supertry/step6_render_score/figures/make_figures.py runs/nzd-supertry/eval_v2
  python nzd-supertry/step6_render_score/figures/make_figures.py runs/nzd-supertry/eval_v2 --only 146-6 --no-render
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.assemble import (  # noqa: E402
    MODELS, REPO, VARIANTS, best_sample, extent_points, ldr_lines, load_captions, load_eval, recolor, rotate,
    variant_text, verdicts, write_ldr,
)

# LDraw colour codes of the highlight style (all standard, so Studio shows them too)
GREY, BLUE, GREEN, ORANGE, RED = 71, 1, 10, 25, 4
VERDICT_CODE = {2: GREEN, 1: ORANGE, 0: RED}
LABELS = {"context": "Input", "reference": "Ground truth", "frozen": "Frozen BrickNet-PT",
          "plain": "Control: plain order", "masked": "Ours: masked order"}
STYLES = ("natural", "highlight")
PITCH, FOV, LDU = -15.0, 15.0, 0.04  # the renderer's defaults; LDraw units to Blender units
PAD_LDU = 20.0  # one stud


def auto_yaw(row, lines):
    """Turn the model so that M faces the camera, 35 degrees off axis. lines: the reference model."""
    pos = np.array([[float(x) for x in ln.split()[2:5]] for ln in lines])
    d = pos[row["n_prefix"]:].mean(0) - pos.mean(0)
    if np.hypot(d[0], d[2]) < 4:  # M sits on the vertical axis of the model: any side shows it
        return 35.0
    return float(np.degrees(np.arctan2(d[2], d[0])) - 125.0)  # the camera looks along +z of the turned model


def camera(points_ldu, yaw, margin):
    """(target in Blender Z-up units, distance) of a camera that frames the points after the model is turned.

    The same framing rule as bricknet_render.orbit_cameras with a fixed target, so every panel of a case can
    be rendered with identical --camera-target / --camera-distance and yaw 0.
    """
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    pad = PAD_LDU * np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    points_ldu = (points_ldu[:, None] + pad[None]).reshape(-1, 3)  # parts reach past their connectors
    p = points_ldu @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
    p = p * [LDU, -LDU, LDU]  # the renderer's Y-up space
    target = (p.min(0) + p.max(0)) / 2
    fwd = np.array([0.0, np.sin(np.radians(PITCH)), np.cos(np.radians(PITCH))])
    right = np.cross([0.0, 1.0, 0.0], -fwd)
    right /= np.linalg.norm(right)
    up = np.cross(-fwd, right)
    tan = np.tan(np.radians(FOV / 2))
    q = p - target
    dist = max((np.abs(q @ right) / tan - q @ fwd).max(), (np.abs(q @ up) / tan - q @ fwd).max()) * margin
    return (target[0], -target[2], target[1]), float(dist)


def render(ldr, png, target, dist, resolution, samples):
    tmp = png.parent / f".render_{png.stem}"
    exe = Path(sys.executable).parent / "bricknet-render"
    env = os.environ | {"BRICKNET_DATA": os.environ.get("BRICKNET_DATA", str(REPO / "baseline/bricknet/data"))}
    cmd = [str(exe), str(ldr), str(tmp), "--views", "1", "--yaw0", "0", "--camera-target", *(f"{x:.4f}" for x in target),
           "--camera-distance", f"{dist:.4f}", "--resolution", f"{resolution}x{resolution}", "--samples", str(samples)]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = tmp / f"{ldr.stem}.png"
    if done.returncode or not out.exists():
        raise RuntimeError(f"render failed for {ldr}:\n{done.stderr[-2000:]}")
    png.parent.mkdir(parents=True, exist_ok=True)
    out.replace(png)
    tmp.rmdir()


def font(size, bold=False):
    names = (["Helvetica.ttc", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"] if bold else ["Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size, index=1 if bold and name.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default(size)


def grid(rows, col_labels, row_labels, cell, notes=None):
    """rows: lists of PNG paths. A white sheet with column headers, row labels on the left, notes under cells."""
    pad, head, side, foot = 10, 46, 30 if any(row_labels) else 0, 26 if notes else 0
    W = side + len(col_labels) * (cell + pad) + pad
    H = head + len(rows) * (cell + foot + pad) + pad
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    for j, text in enumerate(col_labels):
        draw.text((side + pad + j * (cell + pad) + cell / 2, head / 2), text, fill="#1d1d1f", font=font(19, bold=True), anchor="mm")
    for i, row in enumerate(rows):
        y = head + i * (cell + foot + pad)
        if row_labels[i]:
            label = Image.new("RGB", (cell, side), "white")
            ImageDraw.Draw(label).text((cell / 2, side / 2), row_labels[i], fill="#6e6e73", font=font(16), anchor="mm")
            sheet.paste(label.rotate(90, expand=True), (0, y))
        for j, path in enumerate(row):
            x = side + pad + j * (cell + pad)
            sheet.paste(Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS), (x, y))
            if notes and notes[i][j]:
                draw.text((x + cell / 2, y + cell + foot / 2), notes[i][j], fill="#6e6e73", font=font(15), anchor="mm")
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--cases", type=Path, default=HERE / "cases.json")
    ap.add_argument("--out", type=Path, default=HERE, help="where ldr/, panels/ and compare/ go")
    ap.add_argument("--only", nargs="+", help="case ids to (re)do")
    ap.add_argument("--styles", nargs="+", default=list(STYLES), choices=STYLES)
    ap.add_argument("--resolution", type=int, default=900)
    ap.add_argument("--samples", type=int, default=160, help="Cycles samples per pixel")
    ap.add_argument("--cell", type=int, default=420, help="panel size in the comparison sheets, pixels")
    ap.add_argument("--no-render", action="store_true", help="write the LDraw files and rebuild the sheets from existing panels")
    args = ap.parse_args()

    spec = json.loads(args.cases.read_text())
    test, runs = load_eval(args.eval_dir)
    captions = load_captions()
    by_style = {s: [] for s in args.styles}
    for case in spec["cases"]:
        cid, pair = case["id"], case["id"].replace("-", ":")
        row = test[pair]
        pick = {m: case.get("samples", {}).get(m, best_sample(runs[m][pair]["samples"])) for m in MODELS}
        texts = {v: variant_text(row, v, runs.get(v, {}).get(pair), pick.get(v, 0)) for v in VARIANTS}
        lines = {v: ldr_lines(t) for v, t in texts.items()}
        n0 = row["n_prefix"]
        flags = {m: verdicts(row, texts[m]) for m in MODELS}
        yaw = case.get("yaw", auto_yaw(row, lines["reference"]))
        target, dist = camera(np.concatenate([extent_points(t) for t in texts.values()]), yaw, case.get("margin", 1.06))
        notes = [""] * 2 + [f"{sum(f == 2 for f in flags[m])} of {row['k']} parts placed right" for m in MODELS]
        title = f"val model {row['source']}: {captions[row['source']]}"

        for style in args.styles:
            pngs = []
            for n, v in enumerate(VARIANTS):
                codes = [None] * len(lines[v])
                if style == "highlight":
                    added = [BLUE] * row["k"] if v == "reference" else [VERDICT_CODE[f] for f in flags.get(v, [])]
                    codes = [GREY] * n0 + added
                ldr = args.out / "ldr" / cid / style / f"{n + 1}_{v}.ldr"
                write_ldr(ldr, rotate(recolor(lines[v], codes), yaw), f"{title} [{LABELS[v]}, {style}]")
                png = args.out / "panels" / cid / f"{style}_{v}.png"
                if not args.no_render and (not args.only or cid in args.only):
                    render(ldr, png, target, dist, args.resolution, args.samples)
                    print(f"rendered {png.relative_to(args.out)}", flush=True)
                pngs.append(png)
            by_style[style].append((cid, pngs, notes))

        if all((args.out / "panels" / cid / f"{s}_{v}.png").exists() for s in args.styles for v in VARIANTS):
            rows = [pngs for s in args.styles for c, pngs, _ in by_style[s] if c == cid]
            sheet = grid(rows, [LABELS[v] for v in VARIANTS], [s if len(args.styles) > 1 else "" for s in args.styles],
                         args.cell, [notes] + [[""] * len(VARIANTS)] * (len(rows) - 1))
            (args.out / "compare").mkdir(exist_ok=True)
            sheet.save(args.out / "compare" / f"{cid}.png")

    for style, items in by_style.items():
        items = [it for it in items if all(p.exists() for p in it[1])]
        if items:
            grid([p for _, p, _ in items], [LABELS[v] for v in VARIANTS], [""] * len(items), args.cell,
                 [n for _, _, n in items]).save(args.out / "compare" / f"all_{style}.png")
    print(f"sheets in {args.out / 'compare'}")


if __name__ == "__main__":
    main()
