<div align="center">

# BrickNet Render

Rendering for the [BrickNet](https://github.com/kulits/BrickNet) project

</div>

This package turns LDraw assemblies into Blender scenes for rendering using
materials and scene setup adapted from
[Eyesight](https://github.com/kulits/Eyesight).

<div align="center">

<img src="https://raw.githubusercontent.com/kulits/BrickNet-Render/master/docs/images/part_catalog.avif" width="100%">
<img src="https://raw.githubusercontent.com/kulits/BrickNet-Render/master/docs/images/material_catalog.avif" width="100%">

</div>

## Install

```bash
pip install bricknet-render bricknet "bpy>=5.1"
```

Python 3.13+ and Blender/`bpy` 5.1+ are required. If using standalone Blender,
do not install `bpy` separately.

```bash
python -m bricknet_render fetch-glbs
```

Part meshes use BrickNet's user-data directory; set `BRICKNET_DATA` to move it.

## Rendering

```bash
bricknet-render model.ldr out/one
bricknet-render model.ldr out/views --views 8
bricknet-render model.ldr out/hdr --env-map room.exr
```

Add `--save-blend scene.blend` to retain the Blender scene.

## Scripts

```bash
python scripts/render_falling.py scripts/data/car.ldr out/falling
python scripts/color_by_path.py scripts/data/car.ldr out/path
python scripts/render_material_catalog.py out/materials
python scripts/render_part_catalog.py out/parts
```

## Citation

```bibtex
@InProceedings{Kulits_2026_CVPR,
    author    = {Kulits, Peter and Schmid, Cordelia},
    title     = {BrickNet: Graph-Backed Generative Brick Assembly},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {39252-39261}
}
```
