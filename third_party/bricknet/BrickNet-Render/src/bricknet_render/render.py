"""Render LDraw models with materials and multiview scenes."""

import argparse
import hashlib
import math
import re
import shutil
import subprocess
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy
import bmesh
import numpy as np
from mathutils import Color, Euler, Matrix, Vector

from . import __version__
from .data import default_glb_dir
from .lighting import (
    LIGHT_PRESET_NAMES,
    LIGHT_PRESETS,
    LIGHT_RIG_PRESET_NAMES,
    distant_size_to_sun_angle,
)
from .material_catalog import (
    BASE_COLOR_NODE_PROP,
    CATALOG_MATERIAL_KIND,
    CATALOG_MATERIAL_NAME_PROP,
    LDRAW_MATERIAL_NAMES,
    MATERIAL_KIND_PROP,
    MATERIAL_SPECS,
    build_material,
)
from .node_groups import SURFACE_NORMAL_GROUP_NAME, nodegroup_solid_group
from .node_utils import NODE_GROUP_OWNER_PROP

TRANSMISSIVE_AO_RESIDUAL = 0.04
DEFAULT_ENV_EFFECT = 0.2
MAIN_VIEW_LAYER = "Main"
PNG_COMPRESSION = 15
JPEG_QUALITY = 60
DEFAULT_LENS_MM = 28.97056275
BUILT_MATERIAL_PROP = "_bricknet_render_built"
OBJECT_MATERIAL_NAMES_PROP = "_bricknet_render_material_names"
OBJECT_MATERIAL_SURFACES_PROP = "_bricknet_render_material_surfaces"
OBJECT_PRINT_SOURCES_PROP = "_bricknet_render_print_sources"
PART_ID_PROP = "ldraw_part_id"
PART_SOURCE_INDEX_PROP = "ldraw_source_index"
PART_MESH_INDEX_PROP = "ldraw_mesh_index"
TRANSFORM_ATTRIBUTE = "brick_transform"
INSTANCER_PROTOTYPE_PROP = "_bricknet_render_prototype"
PART_STEP_PROP = "ldraw_step"
STEP_VISIBLE_PROP = "visible"
MATERIAL_VARIANT_TEMPLATE_PROP = "_bricknet_render_variant_template"
MATERIAL_VARIANT_RGBA_PROP = "_bricknet_render_variant_rgba"
DIRECT_COLOR_NAME_PROP = "_bricknet_render_direct_color"
MATERIAL_VARIANT_KIND = "variant"
SLOPE_MATERIAL_KIND = "slope"
PRINT_SOURCE_MATERIAL_KIND = "print-source"
PRINT_MATERIAL_KIND = "print"
DIRECT_COLOR_MATERIAL_KIND = "direct-color"
LDRAW_COLOR_MATERIAL_KIND = "ldraw-colour"
GROUND_MATERIAL_KIND = "ground"
VIEW_LAYER_OWNER_PROP = "_bricknet_render_view_layer"
MAIN_VIEW_LAYER_OWNER = "main"
COMPOSITOR_NAME = "Comp_BrickNet Render"
COMPOSITOR_OWNER_PROP = "_bricknet_render_compositor"
RENDER_SCENE_NAME = "BrickNet Render"
RENDER_SCENE_OWNER_PROP = "_bricknet_render_scene"
RENDER_COLLECTION_NAME = "BrickNet Render"
RENDER_COLLECTION_OWNER_PROP = "_bricknet_render_collection"
WORLD_OWNER_PROP = "_bricknet_render_world"
GROUND_NAME = "Ground"
LDRAW_COLOR_DEFINITION_PROP = "_bricknet_render_ldraw_colour"
SLOPE_TEMPLATE_PROP = "_bricknet_render_slope_template"
SLOPE_SCALE_PROP = "_bricknet_render_slope_scale"
SLOPE_GROUP_SOURCE_PROP = "_bricknet_render_slope_group_source"
PRINT_BASE_PROP = "_bricknet_render_print_base"
PRINT_SOURCE_PROP = "_bricknet_render_print_source"
GLTF_SEMANTICS_PROP = "bricknet_render"
PRINT_IMAGE_PROP = "_bricknet_render_print_image"


@dataclass(frozen=True, slots=True)
class PartMaterialSlot:
    """Imported GLB material metadata for one mesh slot."""

    name: str
    surface: str = "base"
    color: int | None = None
    print_material: str | None = None


@dataclass(slots=True)
class PartMeshTemplate:
    """Cached Blender import result for one part GLB."""

    objects: tuple[bpy.types.Object, ...]
    framing_vertices_t: np.ndarray
    raw_center: np.ndarray
    raw_extent: np.ndarray
    slots: tuple[tuple[PartMaterialSlot, ...], ...]


PART_MESH_TEMPLATES: dict[Path, PartMeshTemplate] = {}
CYCLES_DEVICE_CONFIG: tuple[str, str] | None = None
GPU_COMPUTE_ORDER = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")

LDRAW_UNIT_SCALE = 0.04
PART_BOUNDS_INSET_LDU = 0.25
LDRAW_SCALE = np.diag([LDRAW_UNIT_SCALE, LDRAW_UNIT_SCALE, LDRAW_UNIT_SCALE, 1.0])
MIRROR_Y = np.diag([1.0, -1.0, 1.0, 1.0])
Y_TO_Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
Z_TO_Y = np.linalg.inv(Y_TO_Z)
Z_FLIP = np.diag([1.0, 1.0, -1.0, 1.0])
BASE_TRANSFORM = Y_TO_Z @ LDRAW_SCALE @ MIRROR_Y
YUP_BASE = LDRAW_SCALE @ MIRROR_Y


@dataclass(slots=True)
class TurntableOptions:
    """Inputs for an orbit animation."""

    format: str = "png8"
    fps: float = 24.0
    duration: float = 4.0
    rotation_period: float = 4.0
    clockwise: bool = True
    rotate: bool = True
    build_steps: bool = False
    build_duration_ratio: float = 1.0
    falling: bool = False
    temporal_denoise: bool = False
    ffmpeg: str | None = None

    def __post_init__(self) -> None:
        """Normalize the output format and validate animation timing."""

        self.format = {"png": "png8"}.get(self.format.strip().lower(), self.format.strip().lower())
        if self.format not in {"png8", "png16", "mp4", "mov", "gif"}:
            raise ValueError("unsupported turntable format %s" % self.format)
        for name in ("fps", "duration", "rotation_period"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be positive and finite; got %s" % (name, value))
        if not math.isfinite(self.build_duration_ratio) or not 0.0 <= self.build_duration_ratio <= 1.0:
            raise ValueError(
                "build_duration_ratio must be finite and in [0.0, 1.0]; got %s" % self.build_duration_ratio
            )
        if self.falling and not self.build_steps:
            raise ValueError("falling requires build_steps")

    @property
    def frame_end(self) -> int:
        """Return the inclusive final frame for the requested duration."""

        return math.ceil(self.duration * self.fps) - 1


@dataclass(slots=True)
class RenderOptions:
    """Inputs for building and rendering one LDraw scene."""

    ldr_path: str
    glb_dir: str
    out_dir: str
    material: str | None = None
    samples: int = 256
    random_seed: int = 0
    resolution: str = "512x512"
    format: str = "png8"
    light_preset: str = "BUILD_FL"
    light_intensity: float | None = None
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lighting_follows_camera: bool = True
    env_map: str | None = None
    env_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    env_intensity: float = 1.0
    env_projection: str = "EQUIRECTANGULAR"
    env_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    background_mode: str | None = None
    transparent: bool = False
    alpha_adjust: bool = False
    ground_mode: str = "SHADOW"
    ground_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ground_intensity: float = 1.0
    ground_height: float = 0.0
    ground_elevation: float | None = None
    ground_material: str | None = None
    slope_bump: float = 140.0
    postprocess: bool = True
    env_effect: float | None = None
    views: int = 1
    turntable: TurntableOptions | None = None
    yaw0: float = 5.0
    pitch: float = -15.0
    margin: float = 1.05
    field_of_view: float = 15.0
    camera_target: tuple[float, float, float] | None = None
    camera_pan: tuple[float, float] = (0.0, 0.0)
    camera_distance: float | None = None
    camera_type: str = "PERSPECTIVE"
    orthographic_scale: float | None = None
    lens_mm: float | None = None
    fisheye_projection: str = "EQUISOLID"
    fisheye_fov: float = 180.0
    fisheye_lens: float = 10.5
    panorama_latitude: tuple[float, float] = (-90.0, 90.0)
    panorama_longitude: tuple[float, float] = (-180.0, 180.0)
    dof: bool = False
    focus_distance: float = 2.0
    aperture_fstop: float | None = None
    aperture_radius: float | None = None
    aperture_blades: int = 3
    aperture_rotation: float = 0.0
    aperture_ratio: float = 1.0
    device: str = "GPU"
    compute: str = "AUTO"
    adaptive_threshold: float = 0.0
    save_blend: str | None = None
    no_render: bool = False

    def __post_init__(self) -> None:
        """Normalize names and validate render settings."""

        for name in ("samples", "views"):
            if getattr(self, name) <= 0:
                raise ValueError("%s must be positive; got %s" % (name, getattr(self, name)))
        if self.turntable is not None and self.views != 1:
            raise ValueError("turntable animation and multiview output are mutually exclusive")
        for name in ("yaw0", "pitch"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError("%s must be finite; got %s" % (name, getattr(self, name)))
        if not math.isfinite(self.margin) or self.margin <= 0.0:
            raise ValueError("margin must be positive and finite; got %s" % self.margin)
        if not 0.0 < self.field_of_view < 180.0:
            raise ValueError("field_of_view must be in (0.0, 180.0); got %s" % self.field_of_view)
        if not math.isfinite(self.adaptive_threshold) or self.adaptive_threshold < 0.0:
            raise ValueError("adaptive_threshold must be non-negative and finite; got %s" % self.adaptive_threshold)
        if not math.isfinite(self.slope_bump) or self.slope_bump < 0.0:
            raise ValueError("slope_bump must be non-negative and finite; got %s" % self.slope_bump)
        self.light_preset = self.light_preset.strip().upper()
        self.background_mode = None if self.background_mode is None else self.background_mode.strip().upper()
        if (
            self.turntable is not None
            and self.background_mode is None
            and not self.transparent
            and self.ground_mode.strip().upper() == "SHADOW"
            and self.light_preset in LIGHT_RIG_PRESET_NAMES
        ):
            self.background_mode = "TRANSPARENT" if self.turntable.format in {"png8", "png16"} else "FLOOR"
            if self.background_mode == "TRANSPARENT":
                self.ground_mode = "NONE"
        if self.light_intensity is None:
            self.light_intensity = 1.0 if self.background_mode is not None else 0.6
        else:
            self.light_intensity = float(self.light_intensity)
        self.env_effect = (
            (0.1 if self.env_map is not None else DEFAULT_ENV_EFFECT) if self.env_effect is None else self.env_effect
        )
        if not math.isfinite(self.env_effect):
            raise ValueError("env_effect must be finite; got %s" % self.env_effect)
        self.env_effect = max(0.0, self.env_effect)
        self.light_color = tuple(float(c) for c in self.light_color)
        self.light_rotation = tuple(float(angle) for angle in self.light_rotation)
        self.env_color = tuple(float(c) for c in self.env_color)
        self.env_rotation = tuple(float(angle) for angle in self.env_rotation)
        self.ground_color = tuple(float(c) for c in self.ground_color)
        self.env_projection = self.env_projection.strip().upper()
        self.ground_mode = self.ground_mode.strip().upper()
        self.camera_type = self.camera_type.strip().upper()
        self.camera_target = None if self.camera_target is None else tuple(float(value) for value in self.camera_target)
        self.camera_pan = tuple(float(value) for value in self.camera_pan)
        self.fisheye_projection = self.fisheye_projection.strip().upper()
        self.panorama_latitude = tuple(float(angle) for angle in self.panorama_latitude)
        self.panorama_longitude = tuple(float(angle) for angle in self.panorama_longitude)
        if self.aperture_fstop is None and self.aperture_radius is None:
            self.aperture_fstop = 5.6
        if self.background_mode not in {None, "FLOOR", "TRANSPARENT"}:
            raise ValueError("unsupported background_mode %s" % self.background_mode)
        if self.background_mode is not None and self.light_preset not in LIGHT_RIG_PRESET_NAMES:
            raise ValueError("background_mode requires a Build or Mechanic light preset")
        if self.background_mode == "TRANSPARENT" and self.ground_mode == "VISIBLE":
            raise ValueError("TRANSPARENT background_mode requires ground_mode SHADOW or NONE")
        if self.background_mode is not None and self.light_preset in LIGHT_RIG_PRESET_NAMES:
            base_light_intensity = self.light_intensity
            self.env_map = None
            if self.background_mode == "FLOOR":
                self.transparent = False
                self.alpha_adjust = False
                self.ground_mode = "SHADOW"
                self.ground_intensity = 1.0
                self.ground_material = None
                self.env_color = self.ground_color
                self.env_intensity = 1.0
                self.env_effect = 0.15
                self.light_intensity = 0.7 * base_light_intensity
            else:
                self.transparent = True
                self.alpha_adjust = True
                self.env_color = (1.0, 1.0, 1.0)
                self.env_intensity = 0.2 * base_light_intensity
                self.env_effect = 1.0
                self.light_intensity = 0.6 * base_light_intensity
        for name, choices in (
            ("light_preset", LIGHT_PRESETS),
            ("env_projection", {"EQUIRECTANGULAR", "MIRROR_BALL", "LIGHTPROBE"}),
            ("ground_mode", {"NONE", "SHADOW", "VISIBLE"}),
            ("camera_type", {"PERSPECTIVE", "ORTHOGRAPHIC", "FISHEYE", "PANORAMA", "MIRROR_BALL"}),
            ("fisheye_projection", {"EQUIDISTANT", "EQUISOLID"}),
        ):
            if getattr(self, name) not in choices:
                raise ValueError("unsupported %s %s" % (name, getattr(self, name)))

        try:
            width, height = (int(part) for part in self.resolution.strip().lower().split("x"))
        except ValueError as exc:
            raise ValueError("resolution must be WxH; got %s" % self.resolution) from exc
        if width <= 0 or height <= 0:
            raise ValueError("resolution must be positive WxH; got %s" % self.resolution)
        self.resolution = "%dx%d" % (width, height)

        self.format = self.format.strip().lower()
        self.format = {"png": "png8", "jpeg": "jpg"}.get(self.format, self.format)
        if self.format not in {"png8", "png16", "jpg", "bmp", "exr"}:
            raise ValueError("unsupported render format %s" % self.format)
        self.device = self.device.strip().upper()
        if self.device not in {"CPU", "GPU"}:
            raise ValueError("device must be CPU or GPU; got %s" % self.device)
        self.compute = self.compute.strip().upper() or "AUTO"


RGB = tuple[float, float, float]
RGBA = tuple[float, float, float, float]
ColorValue = str | RGB | RGBA


@dataclass(frozen=True, slots=True)
class LDrawMaterialEffect:
    """Parameters of an LDraw ``MATERIAL`` color effect."""

    kind: str
    value: str | None = None
    alpha: int = 255
    luminance: int = 0
    fraction: float | None = None
    volume_fraction: float | None = None
    size: float | None = None
    min_size: float | None = None
    max_size: float | None = None
    fabric: str | None = None


@dataclass(frozen=True, slots=True)
class LDrawColor:
    """One scoped LDraw ``!COLOUR`` definition."""

    name: str
    code: int
    value: str
    edge: str | int
    alpha: int = 255
    luminance: int = 0
    finish: str | None = None
    material: LDrawMaterialEffect | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Placement:
    """One flattened LDraw part placement and its source-order index."""

    stem: str
    color: int
    matrix: np.ndarray
    source_index: int
    local_colors: tuple[LDrawColor, ...] = ()
    step: int = 0


@dataclass(frozen=True, slots=True)
class PlacementSlot:
    """One renderable material slot belonging to an LDraw placement."""

    placement: Placement
    mesh_index: int
    slot_index: int
    gltf_material: str
    material: str
    surface: str = "base"


@dataclass(frozen=True, slots=True)
class MaterialVariant:
    """A recolored derivative of a material template.

    String base colors are sRGB ``#RRGGBB`` values; tuple values are linear
    Blender RGB or RGBA socket values. Alpha does not replace the template's
    transparency model. When ``template`` is omitted, the placement slot's
    original resolved material is retained as the shader template.
    """

    base_color: ColorValue
    template: str | None = None


MaterialMapper = Callable[[PlacementSlot], str | MaterialVariant | None]
LDRAW_COLOR_MATERIALS: dict[str, LDrawColor] = {}


def parse_args(argv: list[str] | None = None, *, prog: str | None = None) -> RenderOptions:
    """Parse renderer arguments."""

    parser = argparse.ArgumentParser(prog=prog, description="BrickNet LDraw renderer")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("ldr_path", help="Path to .ldr file")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument(
        "--glb-dir",
        default=str(default_glb_dir()),
        help="Directory of pre-built .glb parts (default: <BrickNet data dir>/glb)",
    )
    parser.add_argument("--material", default=None, help="Single material, #HEX, or NAME=#HEX@TEMPLATE")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--resolution", default="512x512", help="WxH")
    parser.add_argument(
        "--format",
        default="png8",
        choices=["png", "png8", "png16", "jpg", "jpeg", "bmp", "exr"],
        help="Still-image format (default: PNG8)",
    )
    parser.add_argument("--light-preset", default="BUILD_FL", choices=LIGHT_PRESET_NAMES)
    parser.add_argument("--light-intensity", type=float)
    parser.add_argument("--light-color", type=float, nargs=3, metavar=("R", "G", "B"), default=(1.0, 1.0, 1.0))
    parser.add_argument(
        "--light-rotation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Light-rig rotation in degrees",
    )
    parser.add_argument("--lighting-follows-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-map", default=None)
    parser.add_argument("--env-color", type=float, nargs=3, metavar=("R", "G", "B"), default=(1.0, 1.0, 1.0))
    parser.add_argument("--env-intensity", type=float, default=1.0)
    parser.add_argument(
        "--env-projection",
        choices=["equirectangular", "mirror_ball", "lightprobe"],
        default="equirectangular",
    )
    parser.add_argument(
        "--env-rotation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--background-mode", choices=["floor", "transparent"])
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--alpha-adjust", action="store_true")
    parser.add_argument("--ground-mode", choices=["none", "shadow", "visible"], default="shadow")
    parser.add_argument("--ground-color", type=float, nargs=3, metavar=("R", "G", "B"), default=(1.0, 1.0, 1.0))
    parser.add_argument("--ground-intensity", type=float, default=1.0)
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--ground-elevation", type=float, default=None)
    parser.add_argument("--ground-material", default=None)
    parser.add_argument("--slope-bump", type=float, default=140.0)
    parser.add_argument("--no-postprocess", action="store_false", dest="postprocess")
    parser.add_argument(
        "--env-effect",
        type=float,
        default=None,
        help="Uniform environment fill (default: 0.1 with --env-map, 0.2 otherwise)",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=1,
        help="Number of evenly spaced yaw rotations rendered as still images (default: 1)",
    )
    parser.add_argument("--yaw0", type=float, default=5.0)
    parser.add_argument("--pitch", type=float, default=-15.0)
    parser.add_argument("--margin", type=float, default=1.05)
    parser.add_argument("--field-of-view", type=float, default=15.0)
    parser.add_argument("--camera-target", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-pan", type=float, nargs=2, metavar=("X", "Y"), default=(0.0, 0.0))
    parser.add_argument("--camera-distance", type=float)
    parser.add_argument(
        "--camera-type",
        choices=["perspective", "orthographic", "fisheye", "panorama", "mirror_ball"],
        default="perspective",
    )
    parser.add_argument("--orthographic-scale", type=float)
    parser.add_argument("--lens-mm", type=float)
    parser.add_argument("--fisheye-projection", choices=["equisolid", "equidistant"], default="equisolid")
    parser.add_argument("--fisheye-fov", type=float, default=180.0)
    parser.add_argument("--fisheye-lens", type=float, default=10.5)
    parser.add_argument("--panorama-latitude", type=float, nargs=2, metavar=("MIN", "MAX"), default=(-90.0, 90.0))
    parser.add_argument(
        "--panorama-longitude",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-180.0, 180.0),
    )
    parser.add_argument("--dof", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--focus-distance", type=float, default=2.0)
    parser.add_argument("--aperture-fstop", type=float)
    parser.add_argument("--aperture-radius", type=float, help="Aperture radius in scene meters")
    parser.add_argument("--aperture-blades", type=int, default=3)
    parser.add_argument("--aperture-rotation", type=float, default=0.0)
    parser.add_argument("--aperture-ratio", type=float, default=1.0)
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--compute", default="AUTO")
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=0.0,
        help="Adaptive-sampling noise threshold; 0 disables adaptive sampling",
    )
    parser.add_argument("--save-blend", default=None)
    parser.add_argument("--no-render", action="store_true")
    values = vars(parser.parse_args(argv))
    if values["views"] <= 0:
        parser.error("--views must be positive")
    return RenderOptions(**values)


def material_name(code: int) -> str:
    """Map an LDraw material code to a catalog material name."""

    tag = code >> 24
    if tag in (0x02, 0x03):
        return "LDRAW-0x%08X" % code
    try:
        return LDRAW_MATERIAL_NAMES[code]
    except KeyError:
        raise ValueError("unsupported LDraw color code %d" % code) from None


def parse_ldraw_color(value: str) -> int:
    """Parse a decimal color code or canonical ``0x`` direct color."""

    return int(value, 16) if value.lower().startswith("0x") else int(value, 10)


def _ldraw_rgb(value: str) -> str:
    """Normalize an LDraw RGB token to ``#RRGGBB``."""

    digits = value[1:] if value.startswith("#") else value[2:] if value.lower().startswith("0x") else ""
    if len(digits) != 6:
        raise ValueError("expected #RRGGBB or 0xRRGGBB; got %s" % value)
    try:
        int(digits, 16)
    except ValueError:
        raise ValueError("expected #RRGGBB or 0xRRGGBB; got %s" % value) from None
    return "#" + digits.upper()


def _ldraw_byte(value: str, field: str) -> int:
    """Parse an official integer ALPHA or LUMINANCE value."""

    try:
        result = int(value, 10)
    except ValueError:
        raise ValueError("%s must be an integer byte; got %s" % (field, value)) from None
    if not 0 <= result <= 255:
        raise ValueError("%s must be in [0, 255]; got %s" % (field, value))
    return result


def _ldraw_fraction(value: str, field: str) -> float:
    """Parse a finite material fraction."""

    try:
        result = float(value)
    except ValueError:
        raise ValueError("%s must be a number; got %s" % (field, value)) from None
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("%s must be in (0, 1); got %s" % (field, value))
    return result


def _ldraw_size(value: str, field: str) -> float:
    """Parse a positive finite material particle size."""

    try:
        result = float(value)
    except ValueError:
        raise ValueError("%s must be a number; got %s" % (field, value)) from None
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("%s must be positive and finite; got %s" % (field, value))
    return result


def _parse_ldraw_material(tokens: list[str]) -> LDrawMaterialEffect:
    """Parse the tokens following an LDraw ``MATERIAL`` tag."""

    if not tokens:
        raise ValueError("MATERIAL requires GLITTER, SPECKLE, or FABRIC")
    kind = tokens[0].upper()
    if kind == "FABRIC":
        if len(tokens) > 2 or (len(tokens) == 2 and tokens[1].upper() not in {"VELVET", "CANVAS", "STRING", "FUR"}):
            raise ValueError("invalid FABRIC material")
        return LDrawMaterialEffect(kind=kind, fabric=tokens[1].upper() if len(tokens) == 2 else None)
    if kind not in {"GLITTER", "SPECKLE"}:
        raise ValueError("unsupported LDraw material %s" % tokens[0])

    index = 1

    def take(keyword: str) -> str:
        nonlocal index
        if index + 1 >= len(tokens) or tokens[index].upper() != keyword:
            raise ValueError("%s material requires %s" % (kind, keyword))
        index += 2
        return tokens[index - 1]

    value = _ldraw_rgb(take("VALUE"))
    alpha, luminance = 255, 0
    seen = set()
    while index < len(tokens) and tokens[index].upper() in {"ALPHA", "LUMINANCE"}:
        field = tokens[index].upper()
        if field in seen:
            raise ValueError("duplicate %s material field" % field)
        seen.add(field)
        parsed = _ldraw_byte(take(field), field)
        if field == "ALPHA":
            alpha = parsed
        else:
            luminance = parsed

    fraction = _ldraw_fraction(take("FRACTION"), "FRACTION")
    volume_fraction = _ldraw_fraction(take("VFRACTION"), "VFRACTION") if kind == "GLITTER" else None
    size = min_size = max_size = None
    if index < len(tokens) and tokens[index].upper() == "SIZE":
        size = _ldraw_size(take("SIZE"), "SIZE")
    else:
        min_size = _ldraw_size(take("MINSIZE"), "MINSIZE")
        max_size = _ldraw_size(take("MAXSIZE"), "MAXSIZE")
        if min_size > max_size:
            raise ValueError("MINSIZE must not exceed MAXSIZE")
    if index != len(tokens):
        raise ValueError("unexpected LDraw material field %s" % tokens[index])
    return LDrawMaterialEffect(
        kind=kind,
        value=value,
        alpha=alpha,
        luminance=luminance,
        fraction=fraction,
        volume_fraction=volume_fraction,
        size=size,
        min_size=min_size,
        max_size=max_size,
    )


def parse_ldraw_colour(line: str) -> LDrawColor:
    """Parse one official LDraw ``0 !COLOUR`` directive."""

    tokens = line.split()
    if len(tokens) < 9 or tokens[0] != "0" or tokens[1].upper() != "!COLOUR":
        raise ValueError("expected an LDraw 0 !COLOUR directive")
    name = tokens[2]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", name) is None:
        raise ValueError("invalid LDraw color identifier %s" % name)
    index = 3

    def take(keyword: str) -> str:
        nonlocal index
        if index + 1 >= len(tokens) or tokens[index].upper() != keyword:
            raise ValueError("!COLOUR requires %s" % keyword)
        index += 2
        return tokens[index - 1]

    try:
        code = int(take("CODE"), 10)
    except ValueError:
        raise ValueError("CODE must be a non-negative decimal integer") from None
    if code < 0:
        raise ValueError("CODE must be a non-negative decimal integer")
    value = _ldraw_rgb(take("VALUE"))
    edge_token = take("EDGE")
    if edge_token.startswith("#") or edge_token.lower().startswith("0x"):
        edge: str | int = _ldraw_rgb(edge_token)
    else:
        try:
            edge = int(edge_token, 10)
        except ValueError:
            raise ValueError("EDGE must be an RGB value or decimal color code") from None
        if edge < 0:
            raise ValueError("EDGE must be an RGB value or non-negative color code")

    alpha, luminance = 255, 0
    seen = set()
    while index < len(tokens) and tokens[index].upper() in {"ALPHA", "LUMINANCE"}:
        field = tokens[index].upper()
        if field in seen:
            raise ValueError("duplicate %s color field" % field)
        seen.add(field)
        parsed = _ldraw_byte(take(field), field)
        if field == "ALPHA":
            alpha = parsed
        else:
            luminance = parsed

    finish = material = None
    if index < len(tokens):
        tag = tokens[index].upper()
        if tag in {"CHROME", "PEARLESCENT", "RUBBER", "MATTE_METALLIC", "METAL"}:
            finish = tag
            index += 1
        elif tag == "MATERIAL":
            material = _parse_ldraw_material(tokens[index + 1 :])
            index = len(tokens)
        else:
            raise ValueError("unsupported LDraw finish %s" % tokens[index])
    if index != len(tokens):
        raise ValueError("unexpected LDraw color field %s" % tokens[index])
    return LDrawColor(name, code, value, edge, alpha, luminance, finish, material)


def raw_ldraw_placements(text: str) -> list[Placement]:
    """Read top-level part placements and assign their build-step indices."""

    lines = text.splitlines()
    placements = []
    step = 0
    local_colors: dict[int, LDrawColor] = {}
    for line in lines:
        directive = line.split()
        if len(directive) >= 2 and directive[0] == "0" and directive[1].upper() == "!COLOUR":
            color = parse_ldraw_colour(line)
            local_colors[color.code] = color
            continue
        if len(directive) >= 2 and directive[0] == "0" and directive[1].upper() in {"STEP", "ROTSTEP"}:
            step += 1
            continue
        parts = line.split(maxsplit=14)
        if len(parts) < 15 or parts[0] != "1":
            continue
        values = [float(value) for value in parts[2:14]]
        matrix = np.array(
            [
                [values[3], values[4], values[5], values[0]],
                [values[6], values[7], values[8], values[1]],
                [values[9], values[10], values[11], values[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        placements.append(
            Placement(
                Path(parts[14].replace("\\", "/")).stem.lower(),
                parse_ldraw_color(parts[1]),
                matrix,
                len(placements),
                tuple(local_colors.values()),
                step,
            )
        )
    return placements


def load_placements(path: str, glb_dir: str) -> list[Placement]:
    """Parse top-level placements and check that each part GLB exists."""

    text = Path(path).read_text(encoding="utf-8")
    placements = []
    missing: dict[str, int] = {}
    glb_dir = Path(glb_dir)
    for raw_item in raw_ldraw_placements(text):
        if (glb_dir / ("%s.glb" % raw_item.stem)).exists():
            placements.append(raw_item)
        else:
            missing[raw_item.stem] = missing.get(raw_item.stem, 0) + 1
    if missing:
        summary = ", ".join("%s:%d" % item for item in sorted(missing.items()))
        raise FileNotFoundError("missing GLB(s) for %d placement(s): %s" % (sum(missing.values()), summary))
    return placements


def activate_render_scene() -> bpy.types.Scene:
    """Return and activate the package-owned render scene."""

    owned = [scene for scene in bpy.data.scenes if scene.get(RENDER_SCENE_OWNER_PROP) == RENDER_SCENE_NAME]
    scene = owned[0] if owned else bpy.data.scenes.new(RENDER_SCENE_NAME)
    scene[RENDER_SCENE_OWNER_PROP] = RENDER_SCENE_NAME
    bpy.context.window.scene = scene
    return scene


def render_collection() -> bpy.types.Collection:
    """Return the package-owned collection in the render scene."""

    scene = activate_render_scene()
    owned = [
        collection
        for collection in bpy.data.collections
        if collection.get(RENDER_COLLECTION_OWNER_PROP) == RENDER_COLLECTION_NAME
    ]
    collection = next(
        (collection for collection in owned if scene.collection.children.get(collection.name) is collection),
        owned[0] if owned else None,
    )
    collection = collection or bpy.data.collections.new(RENDER_COLLECTION_NAME)
    collection[RENDER_COLLECTION_OWNER_PROP] = RENDER_COLLECTION_NAME
    if scene.collection.children.get(collection.name) is None:
        scene.collection.children.link(collection)
    return collection


def link_object(obj: bpy.types.Object) -> None:
    """Link an object to the package collection."""

    render_collection().objects.link(obj)


def raw_points_from_imported_object(obj: bpy.types.Object) -> np.ndarray:
    """Return an imported mesh object's vertices in the LDraw part basis."""

    vertices = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", vertices)
    vertices_h = np.ones((len(obj.data.vertices), 4), dtype=np.float64)
    vertices_h[:, :3] = vertices.reshape(-1, 3)
    return (Z_TO_Y @ np.asarray(obj.matrix_world, dtype=np.float64) @ vertices_h.T).T[:, :3]


def _ldraw_color_material_name(color: LDrawColor) -> str:
    """Register and return a deterministic Blender name for a local color."""

    digest = hashlib.sha256(repr(color).encode()).hexdigest()[:10].upper()
    readable = " ".join(color.name.replace("_", " ").split()) or str(color.code)
    name = "LDraw %s [%s]" % (readable[:40], digest)
    previous = LDRAW_COLOR_MATERIALS.setdefault(name, color)
    if previous != color:
        raise RuntimeError("LDraw material name collision for %s" % name)
    return name


def _material_name_for_code(code: int, local_colors: tuple[LDrawColor, ...]) -> str:
    """Resolve one color code against its active local definitions."""

    color = next((color for color in local_colors if color.code == code), None)
    return material_name(code) if color is None else _ldraw_color_material_name(color)


def material_name_from_gltf_slot(
    slot_name: str,
    fallback_color: int,
    local_colors: tuple[LDrawColor, ...] = (),
    color: int | None = None,
) -> str:
    """Map an imported glTF material slot name to a render material name."""

    base = canonical_gltf_material_name(slot_name)
    code = base.split("_", 1)[1] if base.startswith("color_") else ""
    resolved = color if color not in {None, 16} else int(code) if code.isdigit() and code != "16" else fallback_color
    return _material_name_for_code(resolved, local_colors)


@lru_cache(maxsize=4096)
def material_names_from_gltf_slots(
    slots: tuple[PartMaterialSlot, ...],
    fallback_color: int,
    local_colors: tuple[LDrawColor, ...] = (),
) -> tuple[str, ...]:
    """Map one imported GLB slot tuple and LDraw fallback color to render material names."""

    return tuple(material_name_from_gltf_slot(slot.name, fallback_color, local_colors, slot.color) for slot in slots)


def canonical_gltf_material_name(name: str) -> str:
    """Strip only Blender's numeric collision suffix from an imported material name."""

    base, separator, suffix = name.rpartition(".")
    return base if separator and len(suffix) == 3 and suffix.isdigit() else name


def gltf_semantic(element, name: str, default=None):
    """Read one value from the ``bricknet_render`` glTF extras object."""

    semantics = element.get(GLTF_SEMANTICS_PROP)
    return semantics.get(name, default) if hasattr(semantics, "get") else default


def imported_material_slot(material: bpy.types.Material | None) -> PartMaterialSlot:
    """Return the renderer metadata imported with one glTF material."""

    if material is None:
        return PartMaterialSlot("")
    surface = str(gltf_semantic(material, "surface", "base")).strip().lower()
    if surface not in {"base", "slope"}:
        raise ValueError("unsupported GLB material surface %s" % surface)
    color = gltf_semantic(material, "color")
    try:
        color = None if color is None else parse_ldraw_color(str(color))
    except ValueError:
        raise ValueError("invalid GLB material color %s" % color) from None
    printed = gltf_semantic(material, "print", False)
    if not isinstance(printed, bool):
        raise ValueError("invalid GLB material print flag %s" % printed)
    return PartMaterialSlot(
        canonical_gltf_material_name(material.name),
        surface,
        color,
        material.name if printed else None,
    )


def retain_print_material(material: bpy.types.Material) -> None:
    """Keep one imported standard-glTF print material with its images."""

    mark_built_material(material, PRINT_SOURCE_MATERIAL_KIND)
    tree = material.node_tree
    if tree is None:
        raise ValueError("glTF print material %s requires shader nodes" % material.name)
    shader = next(
        (node for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"),
        None,
    )
    if shader is None or not shader.inputs["Base Color"].is_linked or not shader.inputs["Alpha"].is_linked:
        raise ValueError("glTF print material %s requires textured color and alpha" % material.name)
    if tree.nodes.get("Print Source") is None:
        tree.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
        tree.interface.new_socket("Alpha", in_out="OUTPUT", socket_type="NodeSocketFloat")
        output = tree.nodes.new("NodeGroupOutput")
        output.name = "Print Source"
        tree.links.new(shader.inputs["Base Color"].links[0].from_socket, output.inputs["Color"])
        tree.links.new(shader.inputs["Alpha"].links[0].from_socket, output.inputs["Alpha"])
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
            node.image[PRINT_IMAGE_PROP] = True


def import_part_meshes(glb_path: Path) -> list[bpy.types.Object]:
    """Import one part GLB through Blender and return the imported mesh objects."""

    bpy.ops.object.select_all(action="DESELECT")
    debug_value = bpy.app.debug_value
    bpy.app.debug_value = 1 if debug_value == 0 else debug_value
    try:
        result = bpy.ops.import_scene.gltf(filepath=str(glb_path))
    finally:
        bpy.app.debug_value = debug_value
    if "FINISHED" not in result:
        raise RuntimeError("failed to import GLB %s: %s" % (glb_path, result))

    imported = list(bpy.context.selected_objects)
    mesh_objects = sorted((obj for obj in imported if obj.type == "MESH"), key=lambda obj: obj.name)
    matrices = {obj: obj.matrix_world.copy() for obj in mesh_objects}
    for obj in mesh_objects:
        obj.parent = None
        obj.matrix_world = matrices[obj]
    for obj in imported:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    if not mesh_objects:
        raise RuntimeError("empty geometry in %s" % glb_path)
    return mesh_objects


def unlink_object(obj: bpy.types.Object) -> None:
    """Remove an object from all scene collections without deleting its datablock."""

    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)


def auto_smooth_objects(objects: list[bpy.types.Object] | tuple[bpy.types.Object, ...]) -> None:
    """Apply Blender's Smooth by Angle operator to mesh objects."""

    linked_here = []
    states = []
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if not obj.users_collection:
            link_object(obj)
            linked_here.append(obj)
        states.append((obj, obj.hide_viewport, obj.hide_render))
        obj.hide_viewport = False
        obj.hide_render = True
    bpy.context.view_layer.update()
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.shade_auto_smooth(angle=math.radians(30))
    for obj, hide_viewport, hide_render in states:
        obj.select_set(False)
        obj.hide_viewport = hide_viewport
        obj.hide_render = hide_render
    for obj in linked_here:
        unlink_object(obj)


def part_mesh_template(glb_path: Path) -> PartMeshTemplate:
    """Return the cached import template for a part GLB."""

    key = glb_path if glb_path.is_absolute() else glb_path.resolve()
    cached = PART_MESH_TEMPLATES.get(key)
    if cached is not None:
        return cached

    existing_materials = set(bpy.data.materials)
    imported = import_part_meshes(glb_path)
    raw_vertices = np.vstack([raw_points_from_imported_object(obj) for obj in imported])
    if len(raw_vertices) == 0:
        raise RuntimeError("empty geometry in %s" % glb_path)

    slots = []
    imported_materials = {slot.material for obj in imported for slot in obj.material_slots if slot.material is not None}
    for index, obj in enumerate(imported):
        obj.name = ".BrickNet Template %s:%02d" % (glb_path.stem, index)
        obj.data.name = glb_path.stem if len(imported) == 1 else "%s:%02d" % (glb_path.stem, index)
        if not obj.material_slots:
            obj.data.materials.append(None)
            slots.append((PartMaterialSlot("color_16"),))
        else:
            object_slots = tuple(imported_material_slot(slot.material) for slot in obj.material_slots)
            slots.append(object_slots)
            for slot, metadata in zip(obj.material_slots, object_slots):
                if metadata.print_material is not None and slot.material is not None:
                    retain_print_material(slot.material)
            for slot_index in range(len(obj.data.materials)):
                obj.data.materials[slot_index] = None
        obj.data.shade_smooth()
        obj.hide_render = True
        obj.hide_viewport = True
        unlink_object(obj)
    for material in imported_materials:
        if (
            material not in existing_materials
            and material.users == 0
            and material.get(MATERIAL_KIND_PROP) != PRINT_SOURCE_MATERIAL_KIND
        ):
            bpy.data.materials.remove(material)
    objects_without_normals = [obj for obj in imported if not obj.data.has_custom_normals]
    if objects_without_normals:
        auto_smooth_objects(objects_without_normals)

    vertices_t = np.ones((4, len(raw_vertices)), dtype=np.float64)
    vertices_t[:3] = raw_vertices.T
    bmin, bmax = raw_vertices.min(axis=0), raw_vertices.max(axis=0)
    template = PartMeshTemplate(
        tuple(imported),
        vertices_t[:, _hull_indices(raw_vertices)],
        (bmin + bmax) / 2.0,
        bmax - bmin,
        tuple(slots),
    )
    PART_MESH_TEMPLATES[key] = template
    return template


def instantiate_template_object(
    template_obj: bpy.types.Object,
    *,
    name: str,
    matrix_world: Matrix,
    material_names: tuple[str, ...],
    part_id: str,
    source_index: int,
    mesh_index: int,
    step: int,
    surfaces: tuple[str, ...],
    print_sources: tuple[str, ...],
) -> bpy.types.Object:
    """Create one render object from a cached part template."""

    obj = template_obj.copy()
    obj.name = name
    obj.matrix_world = matrix_world
    obj.hide_render = False
    obj.hide_viewport = False
    link_object(obj)
    obj[PART_ID_PROP] = part_id
    obj[PART_SOURCE_INDEX_PROP] = source_index
    obj[PART_MESH_INDEX_PROP] = mesh_index
    obj[PART_STEP_PROP] = step
    obj[OBJECT_MATERIAL_NAMES_PROP] = list(material_names)
    obj[OBJECT_MATERIAL_SURFACES_PROP] = list(surfaces)
    obj[OBJECT_PRINT_SOURCES_PROP] = list(print_sources)
    return obj


def orbit_cameras(
    pts: np.ndarray,
    model_mats: list[np.ndarray],
    look_at: tuple[float, float, float],
    pitch_deg: float,
    yfov_deg: float,
    width: int,
    height: int,
    margin: float,
    *,
    pan: tuple[float, float] = (0.0, 0.0),
    distance: float | None = None,
    auto_center: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Compute tight orbit camera and model matrices in Y-up space."""

    aspect = width / height
    tan_v = math.tan(math.radians(yfov_deg * 0.5))
    tan_h = tan_v * aspect

    cp = math.cos(math.radians(pitch_deg))
    sp = math.sin(math.radians(pitch_deg))
    fwd = np.array([0.0, sp, cp])
    cam_z = -fwd
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, cam_z)
    right /= np.linalg.norm(right)
    up = np.cross(cam_z, right)
    up /= np.linalg.norm(up)

    cam_mats = []
    adj_mats = []
    model_bounds = []
    target = np.array(look_at) - pan[0] * right - pan[1] * up
    target_right = target.dot(right)
    target_up = target.dot(up)
    target_fwd = target.dot(fwd)
    pts_h = np.column_stack((pts, np.ones(len(pts)))).T
    for mat in model_mats:
        frame_pts = (mat @ pts_h).T[:, :3]
        cx = frame_pts.dot(right)
        cy = frame_pts.dot(up)
        cz = frame_pts.dot(fwd)
        dz = cz - target_fwd

        if auto_center:
            dzt = dz * tan_h
            r_max = (cx - dzt).max()
            l_max = (-cx - dzt).max()
            center_x = (r_max - l_max) * 0.5
            fit_distance_h = (r_max + l_max) / (2.0 * tan_h)
            shift_x = center_x - target_right
        else:
            fit_distance_h = (np.abs(cx - target_right) / tan_h - dz).max()
            shift_x = 0.0
        fit_distance_v = (np.abs(cy - target_up) / tan_v - dz).max()
        camera_distance = max(fit_distance_h, fit_distance_v) * margin if distance is None else distance

        adjusted = mat.copy()
        adjusted[:3, 3] += -shift_x * right
        adj_mats.append(adjusted)
        adjusted_pts = frame_pts - shift_x * right
        model_bounds.append((adjusted_pts.min(axis=0), adjusted_pts.max(axis=0)))

        cam = np.eye(4)
        cam[:3, 0] = right
        cam[:3, 1] = up
        cam[:3, 2] = -fwd
        cam[:3, 3] = target - camera_distance * fwd
        cam_mats.append(cam)
    return cam_mats, adj_mats, model_bounds


def clear_scene(*, keep_part_templates: bool = False) -> None:
    """Replace the package-owned scene and release its unused resources."""

    if bpy.context.window is None:
        raise RuntimeError("BrickNet Render requires an active Blender window")
    old_scenes = [scene for scene in bpy.data.scenes if scene.get(RENDER_SCENE_OWNER_PROP) == RENDER_SCENE_NAME]
    factory_scene = None
    if (
        not bpy.data.filepath
        and not bpy.data.is_dirty
        and len(bpy.data.scenes) == 1
        and {(obj.name, obj.type) for obj in bpy.context.scene.objects}
        == {("Cube", "MESH"), ("Camera", "CAMERA"), ("Light", "LIGHT")}
    ):
        factory_scene = bpy.context.scene
        old_scenes.append(factory_scene)
    owned_collections = [
        collection
        for collection in bpy.data.collections
        if collection.get(RENDER_COLLECTION_OWNER_PROP) == RENDER_COLLECTION_NAME
    ]
    if factory_scene is not None:
        owned_collections.extend(factory_scene.collection.children)

    template_objects = {
        obj
        for template in PART_MESH_TEMPLATES.values()
        for obj in template.objects
        if bpy.data.objects.get(obj.name) is obj
    }
    template_meshes = {obj.data for obj in template_objects if obj.data is not None}
    retained_template_objects = template_objects if keep_part_templates else set()
    retained_template_meshes = template_meshes if keep_part_templates else set()

    owned_objects = {
        obj
        for collection in owned_collections
        for obj in collection.objects
        if not any(user_collection not in owned_collections for user_collection in obj.users_collection)
    }
    removed_objects = (owned_objects | template_objects) - retained_template_objects
    removed_data = {obj.data for obj in removed_objects if obj.data is not None} - retained_template_meshes
    removed_data.update(template_meshes - retained_template_meshes)
    removed_actions = {
        obj.animation_data.action
        for obj in removed_objects
        if obj.animation_data is not None and obj.animation_data.action is not None
    }

    new_scene = bpy.data.scenes.new(RENDER_SCENE_NAME)
    new_scene[RENDER_SCENE_OWNER_PROP] = RENDER_SCENE_NAME
    bpy.context.window.scene = new_scene

    for obj in removed_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in owned_collections:
        bpy.data.collections.remove(collection)
    for scene in old_scenes:
        bpy.data.scenes.remove(scene)
    if bpy.data.scenes.get(RENDER_SCENE_NAME) is None:
        new_scene.name = RENDER_SCENE_NAME

    for action in removed_actions:
        if action.users == 0:
            bpy.data.actions.remove(action)
    for data in removed_data:
        if data.users != 0:
            continue
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Camera):
            bpy.data.cameras.remove(data)
        elif isinstance(data, bpy.types.Light):
            bpy.data.lights.remove(data)

    if not keep_part_templates:
        PART_MESH_TEMPLATES.clear()

    for world in list(bpy.data.worlds):
        if world.users == 0 and world.get(WORLD_OWNER_PROP) == RENDER_SCENE_NAME:
            bpy.data.worlds.remove(world)
    transient_kinds = {
        DIRECT_COLOR_MATERIAL_KIND,
        GROUND_MATERIAL_KIND,
        LDRAW_COLOR_MATERIAL_KIND,
        MATERIAL_VARIANT_KIND,
        PRINT_MATERIAL_KIND,
        SLOPE_MATERIAL_KIND,
    }
    if not keep_part_templates:
        transient_kinds.add(PRINT_SOURCE_MATERIAL_KIND)
    for material in list(bpy.data.materials):
        if material.users == 0 and material.get(MATERIAL_KIND_PROP) in transient_kinds:
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users == 0 and image.get(PRINT_IMAGE_PROP):
            bpy.data.images.remove(image)
    LDRAW_COLOR_MATERIALS.clear()
    material_names_from_gltf_slots.cache_clear()
    while groups := [
        group
        for group in bpy.data.node_groups
        if group.users == 0
        and (group.get(COMPOSITOR_OWNER_PROP) == COMPOSITOR_NAME or group.get(NODE_GROUP_OWNER_PROP) is not None)
    ]:
        for group in groups:
            bpy.data.node_groups.remove(group)


def mapped_material_names(
    placement: Placement,
    mesh_index: int,
    slots: tuple[PartMaterialSlot, ...],
    material_names: tuple[str, ...],
    map_material: MaterialMapper | None,
) -> tuple[str, ...]:
    """Apply ``map_material`` to a placement's renderable material slots."""

    if map_material is None:
        return material_names

    def resolve(slot_index: int, gltf_slot: PartMaterialSlot, material: str) -> str:
        slot = PlacementSlot(placement, mesh_index, slot_index, gltf_slot.name, material, gltf_slot.surface)
        choice = map_material(slot)
        if choice is None:
            return material
        if isinstance(choice, str):
            if not choice.strip():
                raise ValueError(
                    "map_material returned a blank name for placement %d mesh %d slot %d"
                    % (placement.source_index, mesh_index, slot_index)
                )
            return choice
        if isinstance(choice, MaterialVariant):
            template = choice.template or material
            variant = build_material_variant(choice.base_color, template)
            if variant is None:
                raise RuntimeError(
                    "material %s not found for placement %d slot %d" % (template, placement.source_index, slot_index)
                )
            return variant.name
        raise TypeError(
            "map_material returned %s for placement %d mesh %d slot %d"
            % (type(choice).__name__, placement.source_index, mesh_index, slot_index)
        )

    return tuple(
        resolve(slot_index, gltf_slot, material)
        for slot_index, (gltf_slot, material) in enumerate(zip(slots, material_names))
    )


def instancer_node_group(prototype: bpy.types.Object) -> bpy.types.NodeTree:
    """Return a node group instancing ``prototype`` with per-point transforms."""

    group = bpy.data.node_groups.new("BrickNet Instancer %s" % prototype.name, "GeometryNodeTree")
    group[NODE_GROUP_OWNER_PROP] = group.name
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes, links = group.nodes, group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    info.transform_space = "ORIGINAL"
    info.inputs["Object"].default_value = prototype
    info.inputs["As Instance"].default_value = True
    on_points = nodes.new("GeometryNodeInstanceOnPoints")
    attribute = nodes.new("GeometryNodeInputNamedAttribute")
    attribute.data_type = "FLOAT4X4"
    attribute.inputs["Name"].default_value = TRANSFORM_ATTRIBUTE
    set_transform = nodes.new("GeometryNodeSetInstanceTransform")

    links.new(group_in.outputs["Geometry"], on_points.inputs["Points"])
    links.new(info.outputs["Geometry"], on_points.inputs["Instance"])
    links.new(on_points.outputs["Instances"], set_transform.inputs["Instances"])
    links.new(attribute.outputs["Attribute"], set_transform.inputs["Transform"])
    links.new(set_transform.outputs["Instances"], group_out.inputs["Geometry"])
    return group


def _hull_indices(points: np.ndarray) -> list[int]:
    """Return the indices of the Nx3 points lying on their convex hull."""

    mesh = bmesh.new()
    for point in points:
        mesh.verts.new(point.tolist())
    mesh.verts.index_update()
    hull = bmesh.ops.convex_hull(mesh, input=mesh.verts, use_existing_faces=False)["geom"]
    keep = sorted(vert.index for vert in hull if isinstance(vert, bmesh.types.BMVert))
    mesh.free()
    # A collinear cloud has no hull; keeping all of it is small and still exact.
    return keep or list(range(len(points)))


def _framing_placements(matrices: np.ndarray) -> np.ndarray:
    """Return the placements whose translation is extreme for their orientation."""

    orientations = matrices[:, :3, :3].round(9).reshape(len(matrices), 9)
    _unique, inverse = np.unique(orientations, axis=0, return_inverse=True)
    keep = []
    for orientation in range(inverse.max() + 1):
        members = np.flatnonzero(inverse == orientation)
        if len(members) > 4:
            members = members[_hull_indices(matrices[members, :3, 3])]
        keep.append(members)
    return matrices[np.concatenate(keep)]


def _instance_matrices(placements: list[Placement], template: PartMeshTemplate) -> np.ndarray:
    """Return the world matrices ``load_model`` would assign each placement."""

    center, extent = template.raw_center, template.raw_extent
    shrink = np.maximum(1.0 - PART_BOUNDS_INSET_LDU / np.where(extent > 0, extent, 1.0), 1e-6)
    matrices = np.stack([placement.matrix for placement in placements])
    inst = Z_FLIP @ matrices @ Z_FLIP
    rotation = inst[:, :3, :3].copy()
    translation = inst[:, :3, 3].copy()
    inst[:, :3, :3] = rotation @ np.diag(shrink)
    inst[:, :3, 3] = translation + rotation @ (center - shrink * center)
    return inst


def _prototype(
    template_obj: bpy.types.Object,
    name: str,
    part_id: str,
    material_names: tuple[str, ...],
    slots: tuple[PartMaterialSlot, ...],
) -> bpy.types.Object:
    """Return a hidden object holding the untransformed geometry for one material set."""

    prototype = template_obj.copy()
    prototype.name = name
    prototype.matrix_world = Matrix.Identity(4)
    prototype.hide_render = True
    prototype.hide_viewport = True
    prototype[PART_ID_PROP] = part_id
    prototype[OBJECT_MATERIAL_NAMES_PROP] = list(material_names)
    prototype[OBJECT_MATERIAL_SURFACES_PROP] = [slot.surface for slot in slots]
    prototype[OBJECT_PRINT_SOURCES_PROP] = [slot.print_material or "" for slot in slots]
    link_object(prototype)
    return prototype


def _instancer(name: str, matrices: np.ndarray, prototype: bpy.types.Object, mesh_index: int) -> bpy.types.Object:
    """Return a point cloud carrying one 4x4 transform per instance."""

    # Set Instance Transform overrides these positions; they remain as the bound_box.
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(matrices))
    mesh.vertices.foreach_set("co", np.ascontiguousarray(matrices[:, :3, 3], dtype=np.float32).ravel())

    attribute = mesh.attributes.new(TRANSFORM_ATTRIBUTE, "FLOAT4X4", "POINT")
    # Blender stores float4x4 column-major; our matrices are row-major.
    attribute.data.foreach_set("value", np.ascontiguousarray(matrices.transpose(0, 2, 1), dtype=np.float32).ravel())

    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj[PART_ID_PROP] = prototype[PART_ID_PROP]
    obj[PART_MESH_INDEX_PROP] = mesh_index
    obj[INSTANCER_PROTOTYPE_PROP] = prototype.name
    modifier = obj.modifiers.new(name="BrickNet Instancer", type="NODES")
    modifier.node_group = instancer_node_group(prototype)
    link_object(obj)
    return obj


def group_placements(
    placements: list[Placement],
    glb_dir: Path,
    map_material: MaterialMapper | None = None,
) -> tuple[dict[tuple[str, tuple], list[Placement]], dict[str, PartMeshTemplate]]:
    """Group placements that can share one instancer, and load their templates."""

    groups: dict[tuple[str, tuple], list[Placement]] = {}
    templates: dict[str, PartMeshTemplate] = {}
    for placement in placements:
        template = templates.get(placement.stem)
        if template is None:
            template = templates[placement.stem] = part_mesh_template(glb_dir / ("%s.glb" % placement.stem))
        materials = tuple(
            mapped_material_names(
                placement,
                mesh_index,
                slots,
                material_names_from_gltf_slots(slots, placement.color, placement.local_colors),
                map_material,
            )
            for mesh_index, slots in enumerate(template.slots)
        )
        groups.setdefault((placement.stem, materials), []).append(placement)
    return groups, templates


def load_model_instanced(
    groups: dict[tuple[str, tuple], list[Placement]],
    templates: dict[str, PartMeshTemplate],
) -> tuple[list[bpy.types.Object], np.ndarray]:
    """Instantiate every placement as Geometry Nodes instances."""

    instancers = []
    yup_points = []
    for index, ((stem, materials), members) in enumerate(sorted(groups.items())):
        template = templates[stem]
        inst = _instance_matrices(members, template)

        framing = YUP_BASE @ _framing_placements(inst) @ template.framing_vertices_t
        yup_points.append(framing.transpose(0, 2, 1).reshape(-1, 4)[:, :3])

        for mesh_index, (template_obj, slots) in enumerate(zip(template.objects, template.slots)):
            label = "%s [%03d]" % (stem, index)
            if len(template.objects) > 1:
                label = "%s:%02d" % (label, mesh_index)
            world = BASE_TRANSFORM @ inst @ (Z_TO_Y @ np.asarray(template_obj.matrix_world, dtype=np.float64))
            prototype = _prototype(template_obj, ".BrickNet Prototype %s" % label, stem, materials[mesh_index], slots)
            instancers.append(_instancer(label, world, prototype, mesh_index))

    return instancers, np.vstack(yup_points)


def load_model(
    args: RenderOptions,
    placements: list[Placement] | None = None,
    map_material: MaterialMapper | None = None,
) -> tuple[list[bpy.types.Object], np.ndarray]:
    """Instantiate each LDraw placement, as objects or as instances."""

    placements = load_placements(args.ldr_path, args.glb_dir) if placements is None else placements
    # A mapper can only split these groups further, never merge them.
    if len({(item.stem, item.color, item.local_colors) for item in placements}) == len(placements):
        return load_model_objects(args, placements, map_material)

    groups, templates = group_placements(placements, Path(args.glb_dir), map_material)
    if len(groups) == len(placements):
        return load_model_objects(args, placements, map_material)

    return load_model_instanced(groups, templates)


def load_model_objects(
    args: RenderOptions,
    placements: list[Placement],
    map_material: MaterialMapper | None = None,
) -> tuple[list[bpy.types.Object], np.ndarray]:
    """Instantiate each LDraw placement from its cached part GLB, one object each."""

    glb_dir = Path(args.glb_dir)

    mesh_objs = []
    yup_points = []

    for placement in placements:
        stem, color = placement.stem, placement.color
        glb_file = glb_dir / ("%s.glb" % stem)
        template = part_mesh_template(glb_file)
        inst = _instance_matrices([placement], template)[0]

        full_transform = BASE_TRANSFORM @ inst

        placement_points = (YUP_BASE @ inst @ template.framing_vertices_t).T[:, :3]
        obj_matrix = Matrix((full_transform @ Z_TO_Y).tolist())

        for mesh_index, (template_obj, slots) in enumerate(zip(template.objects, template.slots)):
            mat_names = mapped_material_names(
                placement,
                mesh_index,
                slots,
                material_names_from_gltf_slots(slots, color, placement.local_colors),
                map_material,
            )
            obj = instantiate_template_object(
                template_obj,
                name=(
                    "%s [%04d]" % (stem, placement.source_index)
                    if len(template.objects) == 1
                    else "%s [%04d]:%02d" % (stem, placement.source_index, mesh_index)
                ),
                matrix_world=obj_matrix @ template_obj.matrix_world,
                material_names=mat_names,
                part_id=stem,
                source_index=placement.source_index,
                mesh_index=mesh_index,
                step=placement.step,
                surfaces=tuple(slot.surface for slot in slots),
                print_sources=tuple(slot.print_material or "" for slot in slots),
            )
            mesh_objs.append(obj)
        yup_points.append(placement_points)

    if not yup_points:
        raise RuntimeError("no geometry loaded")
    return mesh_objs, np.vstack(yup_points)


def orbit_view_matrices(
    args: RenderOptions, points: np.ndarray
) -> tuple[int, list[float], list[np.ndarray], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Return view count, orbit yaws, and Y-up camera/model matrices with model bounds."""

    views = args.views
    res_x, res_y = (int(value) for value in args.resolution.split("x"))

    bmin, bmax = points.min(axis=0), points.max(axis=0)
    center = tuple((bmin + bmax) * 0.5)
    yaws = [args.yaw0 + i * (360.0 / views) for i in range(views)]
    model_rots = []
    for yaw in yaws:
        angle = math.radians(-yaw)
        c, s = math.cos(angle), math.sin(angle)
        cx, _cy, cz = center
        rot = np.eye(4)
        rot[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        rot[:3, 3] = [cx - c * cx - s * cz, 0, cz + s * cx - c * cz]
        model_rots.append(rot)

    target = center if args.camera_target is None else tuple((Z_TO_Y @ np.array((*args.camera_target, 1.0)))[:3])
    explicit_framing = (
        args.camera_target is not None
        or args.camera_pan != (0.0, 0.0)
        or args.camera_distance is not None
        or args.orthographic_scale is not None
    )
    cam_mats, model_mats, model_bounds = orbit_cameras(
        points,
        model_rots,
        target,
        args.pitch,
        args.field_of_view,
        res_x,
        res_y,
        args.margin,
        pan=args.camera_pan,
        distance=args.camera_distance,
        auto_center=not explicit_framing,
    )
    return views, yaws, cam_mats, model_mats, model_bounds


def camera_sensor_dimensions(args: RenderOptions) -> tuple[float, float, float]:
    """Return lens, sensor width, and sensor height in millimeters."""

    lens = args.lens_mm if args.lens_mm is not None and args.lens_mm > 0.0 else DEFAULT_LENS_MM
    width, height = (int(value) for value in args.resolution.split("x"))
    sensor_height = 2.0 * lens * math.tan(math.radians(args.field_of_view / 2.0))
    return lens, sensor_height * width / height, sensor_height


def projection_mask_size(args: RenderOptions) -> tuple[float, float] | None:
    """Return the valid projection ellipse as normalized full width and height."""

    width, height = (int(value) for value in args.resolution.split("x"))
    # The compositor maps 0..1 to the first and last pixel centers; Cycles maps
    # each pixel center into a half-open raster interval.
    pixel_scale = (
        width / (width - 1) if width > 1 else 1.0,
        height / (height - 1) if height > 1 else 1.0,
    )
    if args.camera_type == "MIRROR_BALL" or (
        args.camera_type == "FISHEYE" and args.fisheye_projection == "EQUIDISTANT"
    ):
        return pixel_scale
    if args.camera_type != "FISHEYE":
        return None
    _lens, sensor_width, sensor_height = camera_sensor_dimensions(args)
    radius = 2.0 * args.fisheye_lens * math.sin(math.radians(args.fisheye_fov) / 4.0)
    return (
        2.0 * radius / sensor_width * pixel_scale[0],
        2.0 * radius / sensor_height * pixel_scale[1],
    )


def orbit_camera_data(args: RenderOptions) -> bpy.types.Camera:
    """Create the shared orbit camera data block."""

    cam_data = bpy.data.cameras.new("BrickNet Camera")
    height = int(args.resolution.split("x")[1])
    # Shift the projection center half a pixel downward to align the output raster.
    cam_data.shift_y = -0.5 / height if args.camera_type in {"PERSPECTIVE", "ORTHOGRAPHIC"} else 0.0
    lens, _sensor_width, sensor_height = camera_sensor_dimensions(args)
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_height = sensor_height
    cam_data.lens = lens
    cam_data.clip_end = 10000.0
    if args.camera_type == "ORTHOGRAPHIC":
        cam_data.type = "ORTHO"
        if args.orthographic_scale is not None:
            cam_data.ortho_scale = args.orthographic_scale
    elif args.camera_type == "FISHEYE":
        cam_data.type = "PANO"
        cam_data.panorama_type = "FISHEYE_" + args.fisheye_projection
        cam_data.fisheye_fov = math.radians(args.fisheye_fov)
        cam_data.fisheye_lens = args.fisheye_lens
    elif args.camera_type == "PANORAMA":
        cam_data.type = "PANO"
        cam_data.panorama_type = "EQUIRECTANGULAR"
        cam_data.latitude_min, cam_data.latitude_max = map(math.radians, args.panorama_latitude)
        cam_data.longitude_min, cam_data.longitude_max = map(math.radians, args.panorama_longitude)
    elif args.camera_type == "MIRROR_BALL":
        cam_data.type = "PANO"
        cam_data.panorama_type = "MIRRORBALL"
    fstop = args.aperture_fstop if args.aperture_fstop is not None and args.aperture_fstop > 0.0 else None
    radius = args.aperture_radius if args.aperture_radius is not None and args.aperture_radius > 0.0 else None
    cam_data.dof.use_dof = args.dof and (fstop is not None or radius is not None)
    if fstop is not None:
        cam_data.dof.aperture_fstop = fstop
    elif radius is not None:
        cam_data.dof.aperture_fstop = (
            1.0 / (2.0 * radius) if args.camera_type == "ORTHOGRAPHIC" else cam_data.lens * 1.0e-3 / (2.0 * radius)
        )
    cam_data.dof.aperture_blades = args.aperture_blades
    cam_data.dof.aperture_rotation = math.radians(args.aperture_rotation)
    cam_data.dof.aperture_ratio = args.aperture_ratio
    cam_data.dof.focus_distance = args.focus_distance
    return cam_data


def build_orbit(
    scene: bpy.types.Scene,
    args: RenderOptions,
    mesh_objs: list[bpy.types.Object],
    points: np.ndarray,
    *,
    keyframe_views: bool = True,
) -> list[float]:
    """Create the animated orbit camera and keyframed model rotations."""

    views, yaws, cam_mats, model_mats, _model_bounds = orbit_view_matrices(args, points)
    cams_zup = [Y_TO_Z @ cam for cam in cam_mats]
    models_zup = [Y_TO_Z @ model @ Z_TO_Y for model in model_mats]

    cam_data = orbit_camera_data(args)
    cam_obj = bpy.data.objects.new("BrickNet Camera", cam_data)
    link_object(cam_obj)
    scene.camera = cam_obj
    scene.frame_start = 0
    scene.frame_end = views - 1
    cam_obj.rotation_mode = "QUATERNION"
    for obj in mesh_objs:
        obj.rotation_mode = "QUATERNION"

    base_matrices = {obj.name: obj.matrix_world.copy() for obj in mesh_objs}
    points_h = np.column_stack((points, np.ones(len(points)))).T
    res_x, res_y = (int(value) for value in args.resolution.split("x"))
    for frame in range(views):
        cam_obj.matrix_world = Matrix(cams_zup[frame].tolist())
        if keyframe_views:
            for path in ("location", "rotation_quaternion"):
                cam_obj.keyframe_insert(data_path=path, frame=frame)

        rot_mat = Matrix(models_zup[frame].tolist())
        for obj in mesh_objs:
            obj.matrix_world = rot_mat @ base_matrices[obj.name]
            if keyframe_views:
                for path in ("location", "rotation_quaternion"):
                    obj.keyframe_insert(data_path=path, frame=frame)
        if args.camera_type == "ORTHOGRAPHIC" and args.orthographic_scale is None:
            view_points = (np.linalg.inv(cam_mats[frame]) @ model_mats[frame] @ points_h).T[:, :3]
            width, height = np.ptp(view_points[:, :2], axis=0)
            cam_data.ortho_scale = max(height, width * res_y / res_x) * args.margin
            if keyframe_views:
                cam_data.keyframe_insert(data_path="ortho_scale", frame=frame)

    scene.frame_set(0)
    # From the point cloud: an instancer's bound_box covers its instance origins.
    cam_loc = np.asarray(cam_obj.matrix_world.to_translation(), dtype=np.float64)
    world_points = (models_zup[0] @ Y_TO_Z @ points_h).T[:, :3]
    max_dist = float(np.linalg.norm(world_points - cam_loc, axis=1).max())
    cam_data.clip_end = max(cam_data.clip_end, max_dist * 1.5)
    return yaws


def build_turntable(
    scene: bpy.types.Scene,
    args: RenderOptions,
    mesh_objs: list[bpy.types.Object],
    points: np.ndarray,
) -> list[float]:
    """Create a fixed camera and model animation."""

    turntable = args.turntable
    if turntable is None:
        raise ValueError("turntable configuration is required")
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = np.linalg.norm((points - center)[:, (0, 2)], axis=1).max()
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    circle = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            np.zeros_like(angles),
            center[2] + radius * np.sin(angles),
        )
    )
    framing_points = np.vstack(
        (
            circle + (0.0, points[:, 1].min(), 0.0),
            circle + (0.0, points[:, 1].max(), 0.0),
        )
    )
    build_orbit(scene, args, mesh_objs, framing_points, keyframe_views=False)

    camera_position = scene.camera.matrix_world.to_translation()
    camera_forward = -(scene.camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))
    axis_yup = center if args.camera_target is None else (Z_TO_Y @ np.array((*args.camera_target, 1.0)))[:3]
    axis_zup = Vector((Y_TO_Z @ np.append(axis_yup, 1.0))[:3])
    pivot_zup = camera_position + camera_forward * (axis_zup - camera_position).dot(camera_forward)
    pivot = bpy.data.objects.new("BrickNet Turntable", None)
    pivot.matrix_world = Matrix.Translation(pivot_zup)
    pivot_inverse = pivot.matrix_world.inverted()
    pivot.empty_display_type = "PLAIN_AXES"
    link_object(pivot)
    grouped: dict[int, list[bpy.types.Object]] = {}
    if turntable.build_steps:
        for obj in mesh_objs:
            grouped.setdefault(int(obj.get(PART_STEP_PROP, 0)), []).append(obj)
    else:
        grouped[0] = mesh_objs

    step_parents = []
    for index, (source_step, objects) in enumerate(sorted(grouped.items())):
        parent = pivot
        if turntable.build_steps:
            parent = bpy.data.objects.new("LDraw Step %03d" % (index + 1), None)
            parent[PART_STEP_PROP] = source_step
            parent[STEP_VISIBLE_PROP] = True
            parent.parent = pivot
            parent.empty_display_type = "PLAIN_AXES"
            link_object(parent)
            step_parents.append(parent)
        for obj in objects:
            obj.parent = parent
            obj.matrix_parent_inverse = pivot_inverse
            if turntable.build_steps and not obj.hide_render:
                for path in ("hide_render", "hide_viewport"):
                    driver = obj.driver_add(path).driver
                    variable = driver.variables.new()
                    variable.name = "visible"
                    variable.type = "SINGLE_PROP"
                    variable.targets[0].id = parent
                    variable.targets[0].data_path = '["%s"]' % STEP_VISIBLE_PROP
                    driver.expression = "visible < 0.5"

    # Start one frame before the rendered range so a full cycle has no duplicate endpoint.
    source_frame_zero = -1.0
    end_frame = turntable.duration * turntable.fps + source_frame_zero
    if turntable.rotate:
        end_angle = (
            (-1.0 if turntable.clockwise else 1.0) * turntable.duration / turntable.rotation_period * 2.0 * math.pi
        )
        pivot.rotation_mode = "XYZ"
        pivot.rotation_euler.z = 0.0
        pivot.keyframe_insert(data_path="rotation_euler", index=2, frame=source_frame_zero)
        pivot.rotation_euler.z = end_angle
        pivot.keyframe_insert(data_path="rotation_euler", index=2, frame=end_frame)
        for layer in pivot.animation_data.action.layers:
            for strip in layer.strips:
                for channelbag in strip.channelbags:
                    for fcurve in channelbag.fcurves:
                        for keyframe in fcurve.keyframe_points:
                            keyframe.interpolation = "LINEAR"

    if step_parents:
        step_count = len(step_parents)
        step_tens = step_count // 10
        overlap = max(3, round(math.log(step_tens, 3.0) * 3.0)) if step_tens else 3
        build_duration = math.trunc(turntable.duration * turntable.build_duration_ratio * 100.0) / 100.0
        step_duration = build_duration / (step_count + overlap - 1 if turntable.falling else step_count)
        if turntable.falling:
            magnitude_ldu = np.linalg.norm(np.ptp(points, axis=0)) / LDRAW_UNIT_SCALE
            distance_ldu = max(200.0, math.log(magnitude_ldu / 200.0) * 200.0) if magnitude_ldu > 0.0 else 200.0
            distance = distance_ldu * LDRAW_UNIT_SCALE
            near = max(5.0, distance_ldu / 40.0) * LDRAW_UNIT_SCALE
            scene.camera.data.clip_end += distance

        for index, parent in enumerate(step_parents):
            reveal_frame = index * step_duration * turntable.fps + source_frame_zero
            if reveal_frame > source_frame_zero:
                parent[STEP_VISIBLE_PROP] = False
                parent.keyframe_insert(data_path='["%s"]' % STEP_VISIBLE_PROP, frame=source_frame_zero)
                parent[STEP_VISIBLE_PROP] = True
                parent.keyframe_insert(
                    data_path='["%s"]' % STEP_VISIBLE_PROP,
                    frame=reveal_frame,
                )
            else:
                parent[STEP_VISIBLE_PROP] = True
                parent.keyframe_insert(data_path='["%s"]' % STEP_VISIBLE_PROP, frame=source_frame_zero)
            if turntable.falling:
                end = (index + overlap) * step_duration * turntable.fps + source_frame_zero
                parent.location.z = distance
                parent.keyframe_insert(data_path="location", index=2, frame=reveal_frame)
                parent.location.z = near
                parent.keyframe_insert(
                    data_path="location",
                    index=2,
                    frame=reveal_frame + (end - reveal_frame) * 0.8,
                )
                parent.location.z = 0.0
                parent.keyframe_insert(data_path="location", index=2, frame=end)
            for layer in parent.animation_data.action.layers:
                for strip in layer.strips:
                    for channelbag in strip.channelbags:
                        for fcurve in channelbag.fcurves:
                            interpolation = "LINEAR" if fcurve.data_path == "location" else "CONSTANT"
                            for keyframe in fcurve.keyframe_points:
                                keyframe.interpolation = interpolation

    scene.frame_start = 0
    scene.frame_end = turntable.frame_end
    fps = max(1, round(turntable.fps))
    scene.render.fps = fps
    scene.render.fps_base = fps / turntable.fps
    scene.frame_set(0)
    direction = (1.0 if turntable.clockwise else -1.0) if turntable.rotate else 0.0
    return [
        args.yaw0
        + direction
        * min((frame - source_frame_zero) / turntable.fps, turntable.duration)
        / turntable.rotation_period
        * 360.0
        for frame in range(turntable.frame_end + 1)
    ]


def as_rgba(color: ColorValue) -> RGBA:
    """Coerce hex or RGB(A) color into a Blender RGBA tuple."""

    if isinstance(color, str):
        color = color.removeprefix("#")
        if len(color) != 6:
            raise ValueError("string colors must contain exactly six hexadecimal digits")
        try:
            srgb = Color(tuple(int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4)))
            color = tuple(srgb.from_srgb_to_scene_linear())
        except ValueError:
            raise ValueError("string colors must contain exactly six hexadecimal digits") from None
    channels = tuple(float(value) for value in color)
    if len(channels) not in {3, 4}:
        raise ValueError("tuple colors must contain three or four channels")
    return channels if len(channels) == 4 else (*channels, 1.0)


def material_variant_name(template: str, rgba: RGBA) -> str:
    """Return a deterministic Blender name for a template/color pair."""

    payload = "%s\0%s" % (template, ",".join(float(value).hex() for value in rgba))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12].upper()
    return "CUSTOM-%s__%s" % (digest, template[:38])


def build_material_variant(
    base_color: ColorValue,
    template: str,
    *,
    name: str | None = None,
) -> bpy.types.Material | None:
    """Build or reuse a recolored derivative of ``template``."""

    rgba = as_rgba(base_color)
    if name is not None and (name in MATERIAL_SPECS or name == GROUND_NAME or name.startswith("LDRAW-0x")):
        raise ValueError("custom material name %s is reserved by BrickNet Render" % name)
    name = name or material_variant_name(template, rgba)
    existing = reusable_material(name, MATERIAL_VARIANT_KIND)
    if existing is not None:
        existing_template = existing.get(MATERIAL_VARIANT_TEMPLATE_PROP)
        existing_rgba = existing.get(MATERIAL_VARIANT_RGBA_PROP)
        if existing_template == template and existing_rgba is not None and tuple(existing_rgba) == rgba:
            return existing
        raise RuntimeError("material name %s already identifies a different material" % name)

    placeholder = bpy.data.materials.get(name)
    if placeholder is not None:
        raise RuntimeError("material name %s is already in use" % name)
    template_material = get_or_build_material(template)
    if template_material is None:
        return None
    material = template_material.copy()
    material.name = name
    clear_material_identity(material)
    material[MATERIAL_VARIANT_TEMPLATE_PROP] = template
    material[MATERIAL_VARIANT_RGBA_PROP] = rgba
    recolor_material(material, rgba)
    return mark_built_material(material, MATERIAL_VARIANT_KIND)


def build_custom_material(spec: str) -> bpy.types.Material | None:
    """Build one ``[NAME=]#RRGGBB[@TEMPLATE]`` material."""

    name_part, sep, rest = spec.partition("=")
    name, body = (name_part, rest) if sep else (None, spec)
    hex_code, _sep, template = body.partition("@")
    template = template or "SOLID-WHITE"
    return build_material_variant(hex_code, template, name=name)


def is_custom_material_spec(value: str) -> bool:
    """Return whether ``value`` uses the custom-material CLI grammar."""

    _name, separator, body = value.partition("=")
    return (body if separator else value).startswith("#")


def recolor_material(mat: bpy.types.Material, base_color: ColorValue) -> None:
    """Drive a material's intended base-color inputs to one color."""

    rgba = as_rgba(base_color)
    mat.diffuse_color = rgba
    for node in mat.node_tree.nodes:
        if node.bl_idname == "ShaderNodeRGB" and node.get(BASE_COLOR_NODE_PROP):
            node.outputs[0].default_value = rgba
        elif node.bl_idname == "ShaderNodeGroup":
            for socket in node.inputs:
                if socket.name in {"BaseColor", "Color", "Base Color"} and not socket.is_linked:
                    socket.default_value = rgba


def clear_material_identity(material: bpy.types.Material) -> None:
    """Remove cache identity copied from a source material."""

    for key in (
        BUILT_MATERIAL_PROP,
        MATERIAL_KIND_PROP,
        CATALOG_MATERIAL_NAME_PROP,
        MATERIAL_VARIANT_TEMPLATE_PROP,
        MATERIAL_VARIANT_RGBA_PROP,
        DIRECT_COLOR_NAME_PROP,
        LDRAW_COLOR_DEFINITION_PROP,
        SLOPE_TEMPLATE_PROP,
        SLOPE_SCALE_PROP,
        PRINT_BASE_PROP,
        PRINT_SOURCE_PROP,
    ):
        if key in material:
            del material[key]


def mark_built_material(material: bpy.types.Material, kind: str) -> bpy.types.Material:
    """Tag a material with its cache identity."""

    material[BUILT_MATERIAL_PROP] = True
    material[MATERIAL_KIND_PROP] = kind
    return material


def reusable_material(name: str, kind: str) -> bpy.types.Material | None:
    """Return an already-built material only when its exact kind matches."""

    material = bpy.data.materials.get(name)
    return (
        material
        if material is not None and material.get(BUILT_MATERIAL_PROP) and material.get(MATERIAL_KIND_PROP) == kind
        else None
    )


def get_or_build_catalog_material(name: str) -> bpy.types.Material | None:
    """Build one catalog material once per Blender process."""

    material = build_material(name)
    return None if material is None else mark_built_material(material, CATALOG_MATERIAL_KIND)


def get_or_build_material(name: str) -> bpy.types.Material | None:
    """Build one render material once per Blender process."""

    if name in LDRAW_COLOR_MATERIALS:
        existing = reusable_material(name, LDRAW_COLOR_MATERIAL_KIND)
        if existing is not None:
            return existing
        return build_ldraw_color_material(name)
    if name.startswith("LDRAW-0x"):
        existing = reusable_material(name, DIRECT_COLOR_MATERIAL_KIND)
        if existing is not None:
            return existing
        return build_direct_color_material(name)
    if name in MATERIAL_SPECS:
        return get_or_build_catalog_material(name)
    return reusable_material(name, MATERIAL_VARIANT_KIND)


def ldraw_color_template(color: LDrawColor) -> str:
    """Return the closest neutral material family for a local color."""

    if color.material is not None:
        return {
            "GLITTER": "GLITTER-TRANS_CLEAR",
            "SPECKLE": "SPECKLE-BLACK_SILVER",
            "FABRIC": "CANVAS-WHITE",
        }[color.material.kind]
    if color.finish is not None:
        return {
            "CHROME": "CHROME-SILVER",
            "PEARLESCENT": "PEARL-WHITE",
            "RUBBER": "RUBBER-TRANS_CLEAR" if color.alpha < 255 else "RUBBER-WHITE",
            "MATTE_METALLIC": "PEARL-FLAT_SILVER",
            "METAL": "METAL-SILVER",
        }[color.finish]
    return "TRANS-CLEAR" if color.alpha < 255 else "SOLID-WHITE"


def _set_ldraw_secondary_color(material: bpy.types.Material, effect: LDrawMaterialEffect) -> None:
    """Set the shared procedural controls represented by an LDraw effect."""

    label = {"GLITTER": "RGB_Chip", "SPECKLE": "RGB_Second"}.get(effect.kind)
    if label is None or effect.value is None:
        return
    rgba = as_rgba(effect.value)
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeRGB" and node.label == label:
            node.outputs["Color"].default_value = rgba
    group_node = next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeGroup")
    group_node.inputs["Chip Alpha"].default_value = effect.alpha / 255.0
    group_node.inputs["Chip Luminance"].default_value = effect.luminance / 255.0
    group_node.inputs["Minimum Particle Size"].default_value = (
        effect.size if effect.size is not None else effect.min_size
    )
    group_node.inputs["Maximum Particle Size"].default_value = (
        effect.size if effect.size is not None else effect.max_size
    )
    group_node.inputs["Chip Fraction"].default_value = effect.fraction
    group_node.inputs["LDraw Distribution"].default_value = 1.0


def _add_ldraw_luminance(material: bpy.types.Material, color: str, luminance: int) -> None:
    """Add the self-emission declared by an LDraw color."""

    if luminance == 0:
        return
    tree = material.node_tree
    output = next(node for node in tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    surface = output.inputs["Surface"]
    source = surface.links[0].from_socket
    tree.links.remove(surface.links[0])
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "LDraw Luminance"
    emission.inputs["Color"].default_value = as_rgba(color)
    emission.inputs["Strength"].default_value = luminance / 255.0
    add = tree.nodes.new("ShaderNodeAddShader")
    add.name = "LDraw Luminance Surface"
    tree.links.new(source, add.inputs[0])
    tree.links.new(emission.outputs["Emission"], add.inputs[1])
    tree.links.new(add.outputs["Shader"], surface)


def _apply_ldraw_alpha(material: bpy.types.Material, alpha: int) -> None:
    """Mix a material surface with transparency using the official alpha byte."""

    material.diffuse_color = (*material.diffuse_color[:3], alpha / 255.0)
    if alpha == 255:
        return
    tree = material.node_tree
    output = next(node for node in tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    volume = output.inputs["Volume"]
    if volume.is_linked:
        source = volume.links[0].from_socket
        tree.links.remove(volume.links[0])
        empty = tree.nodes.new("ShaderNodeVolumeAbsorption")
        empty.name = "Empty LDraw Volume"
        empty.inputs["Density"].default_value = 0.0
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.name = "LDraw Volume Alpha"
        mix.inputs["Fac"].default_value = alpha / 255.0
        tree.links.new(empty.outputs["Volume"], mix.inputs[1])
        tree.links.new(source, mix.inputs[2])
        tree.links.new(mix.outputs["Shader"], volume)
    surface = output.inputs["Surface"]
    source = surface.links[0].from_socket
    tree.links.remove(surface.links[0])
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.name = "LDraw Transparency"
    if alpha == 0:
        tree.links.new(transparent.outputs["BSDF"], surface)
        return
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "LDraw Alpha"
    mix.inputs["Fac"].default_value = alpha / 255.0
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(source, mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], surface)


def _ldraw_color_properties(color: LDrawColor) -> dict:
    """Return portable Blender custom properties for a local color."""

    return asdict(color)


def build_ldraw_color_material(name: str) -> bpy.types.Material:
    """Build a native Blender material from a scoped LDraw color."""

    color = LDRAW_COLOR_MATERIALS[name]
    placeholder = bpy.data.materials.get(name)
    if placeholder is not None:
        raise RuntimeError("material name %s is already in use" % name)
    template = get_or_build_catalog_material(ldraw_color_template(color))
    material = template.copy()
    material.name = name
    clear_material_identity(material)
    recolor_material(material, color.value)
    if color.material is not None:
        _set_ldraw_secondary_color(material, color.material)
    _add_ldraw_luminance(material, color.value, color.luminance)
    _apply_ldraw_alpha(material, color.alpha)
    material[LDRAW_COLOR_DEFINITION_PROP] = _ldraw_color_properties(color)
    return mark_built_material(material, LDRAW_COLOR_MATERIAL_KIND)


def build_direct_color_material(name: str) -> bpy.types.Material:
    """Build one opaque or translucent LDraw direct-color material."""

    placeholder = bpy.data.materials.get(name)
    if placeholder is not None:
        raise RuntimeError("material name %s is already in use" % name)
    code = int(name.removeprefix("LDRAW-"), 16)
    tag = code >> 24
    if tag not in {0x02, 0x03}:
        raise ValueError("unsupported LDraw direct-color tag in %s" % name)
    color = "#%06X" % (code & 0xFFFFFF)
    template = "TRANS-CLEAR" if tag == 0x03 else "SOLID-WHITE"
    template_material = get_or_build_catalog_material(template)
    mat = template_material.copy()
    mat.name = name
    clear_material_identity(mat)
    mat[DIRECT_COLOR_NAME_PROP] = name
    recolor_material(mat, color)
    return mark_built_material(mat, DIRECT_COLOR_MATERIAL_KIND)


def _slope_datablock_name(name: str, scale: float) -> str:
    """Return a readable Blender name for one slope variant."""

    scale_name = str(int(scale)) if float(scale).is_integer() else repr(float(scale))
    suffix = " (Slope %s)" % scale_name
    return name[: 63 - len(suffix)] + suffix


def _uses_surface_normal(tree: bpy.types.NodeTree, memo: dict[int, bool]) -> bool:
    """Return whether a shader tree reaches the shared surface-normal group."""

    key = tree.as_pointer()
    if key not in memo:
        memo[key] = any(
            node.bl_idname == "ShaderNodeGroup"
            and node.node_tree is not None
            and (node.node_tree.name == SURFACE_NORMAL_GROUP_NAME or _uses_surface_normal(node.node_tree, memo))
            for node in tree.nodes
        )
    return memo[key]


def _slope_group_variant(
    source: bpy.types.NodeTree,
    scale: float,
    variants: dict[int, bpy.types.NodeTree],
    surface_normal: dict[int, bool],
) -> bpy.types.NodeTree:
    """Copy one shader-group path and add slope normals at its leaves."""

    key = source.as_pointer()
    if key in variants:
        return variants[key]
    if not _uses_surface_normal(source, surface_normal):
        return source

    name = _slope_datablock_name(source.name, scale)
    existing = bpy.data.node_groups.get(name)
    if existing is not None:
        if (
            existing.get(NODE_GROUP_OWNER_PROP) == name
            and existing.get(SLOPE_GROUP_SOURCE_PROP) == source.name
            and existing.get(SLOPE_SCALE_PROP) == scale
        ):
            variants[key] = existing
            return existing
        raise RuntimeError("node-group name %s is already in use" % name)

    variant = source.copy()
    variant.name = name
    variant[NODE_GROUP_OWNER_PROP] = name
    variant[SLOPE_GROUP_SOURCE_PROP] = source.name
    variant[SLOPE_SCALE_PROP] = scale
    variants[key] = variant
    _instrument_slope_tree(variant, scale, variants, surface_normal)
    return variant


def _instrument_slope_tree(
    tree: bpy.types.NodeTree,
    scale: float,
    variants: dict[int, bpy.types.NodeTree],
    surface_normal: dict[int, bool],
) -> None:
    """Replace surface-normal paths in a shader tree with slope variants."""

    for node in tuple(tree.nodes):
        if node.bl_idname != "ShaderNodeGroup" or node.node_tree is None:
            continue
        if node.node_tree.name == SURFACE_NORMAL_GROUP_NAME:
            node.inputs["Rough Surface Scale"].default_value = scale
        else:
            node.node_tree = _slope_group_variant(node.node_tree, scale, variants, surface_normal)


def build_slope_material(template: str, scale: float) -> bpy.types.Material | None:
    """Build or reuse a rough-surface variant of one material."""

    base = get_or_build_material(template)
    if base is None or scale == 0.0:
        return base
    name = _slope_datablock_name(base.name, scale)
    existing = reusable_material(name, SLOPE_MATERIAL_KIND)
    if existing is not None:
        if existing.get(SLOPE_TEMPLATE_PROP) == base.name and existing.get(SLOPE_SCALE_PROP) == scale:
            return existing
        raise RuntimeError("material name %s already identifies a different slope material" % name)
    if bpy.data.materials.get(name) is not None:
        raise RuntimeError("material name %s is already in use" % name)

    material = base.copy()
    material.name = name
    clear_material_identity(material)
    material[SLOPE_TEMPLATE_PROP] = base.name
    material[SLOPE_SCALE_PROP] = scale
    _instrument_slope_tree(material.node_tree, scale, {}, {})
    return mark_built_material(material, SLOPE_MATERIAL_KIND)


def build_print_material(base: bpy.types.Material, source_name: str) -> bpy.types.Material:
    """Build a printed surface over one selected material."""

    source = reusable_material(source_name, PRINT_SOURCE_MATERIAL_KIND)
    if source is None or source.node_tree is None:
        raise RuntimeError("glTF print material %s not found" % source_name)

    name = "%s (Print %s)" % (base.name, source.name)
    existing = reusable_material(name, PRINT_MATERIAL_KIND)
    if existing is not None:
        if existing.get(PRINT_BASE_PROP) == base.name and existing.get(PRINT_SOURCE_PROP) == source.name:
            return existing
        raise RuntimeError("material name %s already identifies a different print" % name)
    if bpy.data.materials.get(name) is not None:
        raise RuntimeError("material name %s is already in use" % name)

    material = base.copy()
    material.name = name
    clear_material_identity(material)
    material[PRINT_BASE_PROP] = base.name
    material[PRINT_SOURCE_PROP] = source.name
    tree = material.node_tree
    output = next(node for node in tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial")
    target = output.inputs["Surface"]
    base_shader = target.links[0].from_socket
    tree.links.remove(target.links[0])

    texture = tree.nodes.new("ShaderNodeGroup")
    texture.name = "Print Texture"
    texture.label = source.name
    texture.node_tree = source.node_tree
    sticker = tree.nodes.new("ShaderNodeGroup")
    sticker.name = "Printed Solid"
    sticker.node_tree = (
        _slope_group_variant(nodegroup_solid_group(), base[SLOPE_SCALE_PROP], {}, {})
        if base.get(MATERIAL_KIND_PROP) == SLOPE_MATERIAL_KIND
        else nodegroup_solid_group()
    )
    tree.links.new(texture.outputs["Color"], sticker.inputs["BaseColor"])
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "Print over Base Material"
    tree.links.new(texture.outputs["Alpha"], mix.inputs["Fac"])
    tree.links.new(base_shader, mix.inputs[1])
    tree.links.new(sticker.outputs["Shader"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], target)
    return mark_built_material(material, PRINT_MATERIAL_KIND)


def semantic_material(
    name: str,
    surface: str,
    slope_bump: float,
    print_source: str = "",
) -> bpy.types.Material | None:
    """Return the material selected by one GLB surface tag."""

    material = build_slope_material(name, slope_bump) if surface == "slope" else get_or_build_material(name)
    return build_print_material(material, print_source) if material is not None and print_source else material


def build_materials(args: RenderOptions, mesh_objs: list[bpy.types.Object]) -> tuple[str, ...]:
    """Build and assign BrickNet materials."""

    mesh_objs = [bpy.data.objects[name] if (name := obj.get(INSTANCER_PROTOTYPE_PROP)) else obj for obj in mesh_objs]
    assignments = [
        (obj, index, name, surface, source)
        for obj in mesh_objs
        for index, (name, surface, source) in enumerate(
            zip(
                obj[OBJECT_MATERIAL_NAMES_PROP],
                obj.get(OBJECT_MATERIAL_SURFACES_PROP, ["base"] * len(obj[OBJECT_MATERIAL_NAMES_PROP])),
                obj.get(OBJECT_PRINT_SOURCES_PROP, [""] * len(obj[OBJECT_MATERIAL_NAMES_PROP])),
            )
        )
    ]
    if args.material:
        base = (
            build_custom_material(args.material)
            if is_custom_material_spec(args.material)
            else get_or_build_material(args.material)
        )
        if base is None:
            raise RuntimeError("material %s not found" % args.material)
        created = {
            (name, surface, source): semantic_material(base.name, surface, args.slope_bump, source)
            for _obj, _index, name, surface, source in assignments
        }
        for obj, index, name, surface, source in assignments:
            obj.material_slots[index].link = "OBJECT"
            obj.material_slots[index].material = created[name, surface, source]
        for obj in mesh_objs:
            obj.color = obj.material_slots[0].material.diffuse_color
        return tuple(sorted({material.name for material in created.values()}))

    created = {}
    for name, surface, source in sorted(
        {(name, surface, source) for _obj, _index, name, surface, source in assignments}
    ):
        material = semantic_material(name, surface, args.slope_bump, source)
        if material is None:
            raise RuntimeError("material %s not found" % name)
        created[name, surface, source] = material
    for obj, index, name, surface, source in assignments:
        material = created[name, surface, source]
        obj.material_slots[index].link = "OBJECT"
        obj.material_slots[index].material = material
    for obj in mesh_objs:
        obj.color = obj.material_slots[0].material.diffuse_color
    return tuple(sorted({material.name for material in created.values()}))


def mesh_world_bounds(mesh_objs: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    """Return exact world-space bounds over all mesh vertices."""

    mins, maxs = [], []
    for obj in mesh_objs:
        prototype = obj.get(INSTANCER_PROTOTYPE_PROP)
        source = bpy.data.objects[prototype] if prototype else obj
        if not source.data.vertices:
            continue
        vertices = np.empty(len(source.data.vertices) * 3, dtype=np.float32)
        source.data.vertices.foreach_get("co", vertices)
        local = vertices.reshape(-1, 3).T.astype(np.float64)

        placement = np.asarray(obj.matrix_world, dtype=np.float64)
        if not prototype:
            world = (placement[:3, :3] @ local).T + placement[:3, 3]
            mins.append(world.min(axis=0))
            maxs.append(world.max(axis=0))
            continue

        matrices = np.empty(len(obj.data.attributes[TRANSFORM_ATTRIBUTE].data) * 16, dtype=np.float32)
        obj.data.attributes[TRANSFORM_ATTRIBUTE].data.foreach_get("value", matrices)
        matrices = matrices.astype(np.float64)
        # Blender stores float4x4 column-major; our matrices are row-major.
        matrices = placement @ matrices.reshape(-1, 4, 4).transpose(0, 2, 1)
        unique, inverse = np.unique(matrices[:, :3, :3].reshape(-1, 3).round(9), axis=0, return_inverse=True)
        support = unique @ local
        offsets = matrices[:, :3, 3]
        mins.append((support.min(axis=1)[inverse].reshape(-1, 3) + offsets).min(axis=0))
        maxs.append((support.max(axis=1)[inverse].reshape(-1, 3) + offsets).max(axis=0))
    if not mins:
        raise RuntimeError("no geometry to measure")
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def add_ground(
    args: RenderOptions,
    mesh_objs: list[bpy.types.Object],
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> bpy.types.Object | None:
    """Add the configured visible or shadow-catcher ground plane."""

    if args.ground_mode == "NONE" or (bounds is None and not mesh_objs):
        return None
    bmin, bmax = mesh_world_bounds(mesh_objs) if bounds is None else bounds
    min_x, min_y, min_z = bmin
    max_x, max_y, _max_z = bmax
    extent = max(max_x - min_x, max_y - min_y)
    half = max(extent * 200.0, 1.0)
    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    gz = args.ground_elevation - 0.0001 if args.ground_elevation is not None else min_z - args.ground_height - 0.0001

    mesh = bpy.data.meshes.new(name=GROUND_NAME)
    mesh.from_pydata(
        [
            (cx - half, cy - half, gz),
            (cx + half, cy - half, gz),
            (cx + half, cy + half, gz),
            (cx - half, cy + half, gz),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    ground = bpy.data.objects.new(GROUND_NAME, mesh)
    link_object(ground)
    ground.data.materials.append(ground_material(args))

    if args.ground_mode == "SHADOW":
        ground.is_shadow_catcher = True
        ground.visible_shadow = False
        ground.visible_diffuse = False
        ground.visible_glossy = False
        ground.visible_transmission = False
        ground.visible_volume_scatter = False
    return ground


def ground_material(args: RenderOptions) -> bpy.types.Material:
    """Return the configured ground material."""

    if args.ground_material is not None:
        material = get_or_build_material(args.ground_material)
        if material is None:
            raise ValueError("unknown ground material %s" % args.ground_material)
        return material
    name = GROUND_NAME
    existing = bpy.data.materials.get(name)
    if existing is not None:
        if existing.get(MATERIAL_KIND_PROP) != GROUND_MATERIAL_KIND:
            raise RuntimeError("material name %s is already in use" % name)
        return existing

    mat = bpy.data.materials.new(name=name)
    mat[MATERIAL_KIND_PROP] = GROUND_MATERIAL_KIND
    color = _scaled_color(args.ground_color, args.ground_intensity, normalize=True)
    mat.diffuse_color = (*color, 1.0)
    tree = mat.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (*color, 1.0)
    tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _scaled_color(
    color: tuple[float, float, float],
    intensity: float,
    *,
    normalize: bool = False,
) -> tuple[float, float, float]:
    """Scale non-negative RGB, optionally limiting its peak to one."""

    scaled = tuple(max(0.0, c * max(0.0, intensity)) for c in color)
    divisor = max(1.0, *scaled) if normalize else 1.0
    return tuple(c / divisor for c in scaled)


def _resolved_light_color(args: RenderOptions) -> tuple[float, float, float]:
    """Clamp the light tint, using white when every channel is negative."""

    return tuple(max(0.0, c) for c in args.light_color) if any(c >= 0.0 for c in args.light_color) else (1.0, 1.0, 1.0)


def ambient_light_radiance(args: RenderOptions) -> tuple[float, float, float]:
    """Return the radiance contributed by ambient lights in the selected rig."""

    color = _resolved_light_color(args)
    return tuple(
        max(0.0, args.light_intensity)
        * color[c]
        * sum(spec.color[c] * spec.strength for spec in LIGHT_PRESETS[args.light_preset] if spec.type == "ambient")
        for c in range(3)
    )


def environment_source(tree: bpy.types.NodeTree, args: RenderOptions, yaws: list[float]):
    """Build the uniform-color or image environment source."""

    tint = _scaled_color(args.env_color, args.env_intensity if args.env_map is None else 1.0)
    if args.env_map is None:
        return (*tint, 1.0)

    path = Path(args.env_map).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("environment map not found: %s" % path)
    image = bpy.data.images.load(str(path), check_existing=True)
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Environment Coordinates"
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.name = "Environment Rotation"
    tree.links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    rotations = [
        Euler(tuple(math.radians(angle) for angle in resolved_environment_rotation(args, yaw)), "XYZ") for yaw in yaws
    ]
    mapping.inputs["Rotation"].default_value = rotations[0]
    if len(rotations) > 1 and not args.lighting_follows_camera:
        for frame, rotation in enumerate(rotations):
            mapping.inputs["Rotation"].default_value = rotation
            mapping.inputs["Rotation"].keyframe_insert(data_path="default_value", frame=frame)
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)

    if args.env_projection == "LIGHTPROBE":

        def math_node(operation, left, right=None):
            node = tree.nodes.new("ShaderNodeMath")
            node.operation = operation
            if isinstance(left, (int, float)):
                node.inputs[0].default_value = left
            else:
                tree.links.new(left, node.inputs[0])
            if right is not None:
                if isinstance(right, (int, float)):
                    node.inputs[1].default_value = right
                else:
                    tree.links.new(right, node.inputs[1])
            return node.outputs[0]

        normalize = tree.nodes.new("ShaderNodeVectorMath")
        normalize.name = "Normalize Lightprobe Direction"
        normalize.operation = "NORMALIZE"
        tree.links.new(mapping.outputs["Vector"], normalize.inputs[0])
        split = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(normalize.outputs["Vector"], split.inputs["Vector"])
        x2 = math_node("MULTIPLY", split.outputs["X"], split.outputs["X"])
        z2 = math_node("MULTIPLY", split.outputs["Z"], split.outputs["Z"])
        d = math_node("SQRT", math_node("ADD", x2, z2))
        d_nonzero = math_node("GREATER_THAN", d, 1.0e-8)
        r = math_node(
            "MULTIPLY",
            math_node(
                "DIVIDE",
                math_node("ARCCOSINE", split.outputs["Y"]),
                math_node("MULTIPLY", math_node("MAXIMUM", d, 1.0e-8), 2.0 * math.pi),
            ),
            d_nonzero,
        )
        combine = tree.nodes.new("ShaderNodeCombineXYZ")
        tree.links.new(math_node("ADD", 0.5, math_node("MULTIPLY", split.outputs["X"], r)), combine.inputs["X"])
        tree.links.new(math_node("ADD", 0.5, math_node("MULTIPLY", split.outputs["Z"], r)), combine.inputs["Y"])
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.name = "Environment Lightprobe"
        texture.image = image
        texture.extension = "EXTEND"
        tree.links.new(combine.outputs["Vector"], texture.inputs["Vector"])
    else:
        texture = tree.nodes.new("ShaderNodeTexEnvironment")
        texture.name = "Environment Map"
        texture.image = image
        texture.projection = args.env_projection
        tree.links.new(mapping.outputs["Vector"], texture.inputs["Vector"])

    multiply = tree.nodes.new("ShaderNodeMix")
    multiply.name = "Environment Tint and Intensity"
    multiply.data_type = "RGBA"
    multiply.blend_type = "MULTIPLY"
    multiply.inputs["Factor"].default_value = 1.0
    multiply.inputs["B"].default_value = (*tint, 1.0)
    tree.links.new(texture.outputs["Color"], multiply.inputs["A"])
    return multiply.outputs["Result"]


def resolved_environment_rotation(args: RenderOptions, yaw: float = 0.0) -> tuple[float, float, float]:
    """Return the Blender XYZ rotation for the selected environment."""

    if yaw == 0.0 or args.lighting_follows_camera:
        return args.env_rotation
    source_rotation = Euler(tuple(math.radians(angle) for angle in args.env_rotation), "XYZ").to_matrix()
    return tuple(
        math.degrees(angle) for angle in (source_rotation @ Matrix.Rotation(math.radians(yaw), 3, "Z")).to_euler("XYZ")
    )


def _camera_light_yaw(yaw: float, pitch: float) -> float:
    """Return the camera-following preset yaw in degrees."""

    y, p = math.radians(yaw), math.radians(pitch)
    x, z = math.sin(y) * math.cos(p), -math.cos(y)
    return 0.0 if math.hypot(x, z) < 1.0e-12 else -math.degrees(math.atan2(x, z))


def resolved_light_rotation(args: RenderOptions, yaw: float) -> Euler:
    """Return the light-rig rotation for one fixed-camera model view."""

    rotation = list(args.light_rotation)
    if args.lighting_follows_camera:
        rotation[2] += _camera_light_yaw(yaw, args.pitch)
    source = Euler(tuple(math.radians(angle) for angle in rotation), "XYZ").to_matrix()
    return (Matrix.Rotation(math.radians(-yaw), 3, "Z") @ source).to_euler("XYZ")


def configure_distant_light_shader(
    light: bpy.types.SunLight,
    color: tuple[float, float, float],
    strength: float,
    size: float,
) -> None:
    """Configure a distant light with disk-profiled emission."""

    tree = light.node_tree
    emission = tree.nodes["Emission"]
    emission.inputs["Color"].default_value = (*color, 1.0)
    if size == 0.0:
        emission.inputs["Strength"].default_value = strength
        return

    # Use a tangent-disk distribution: p(theta) is proportional to cos(theta)^-3.
    # Blender's Sun samples its cone uniformly, so its emission needs that profile.
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Distant Ray Direction"
    axis = tree.nodes.new("ShaderNodeVectorTransform")
    axis.name = "Distant Light Axis"
    axis.vector_type = "VECTOR"
    axis.convert_from = "OBJECT"
    axis.convert_to = "WORLD"
    axis.inputs["Vector"].default_value = (0.0, 0.0, 1.0)
    normalized_axis = tree.nodes.new("ShaderNodeVectorMath")
    normalized_axis.name = "Normalized Distant Light Axis"
    normalized_axis.operation = "NORMALIZE"
    tree.links.new(axis.outputs["Vector"], normalized_axis.inputs[0])
    cosine = tree.nodes.new("ShaderNodeVectorMath")
    cosine.name = "Distant Ray Cosine"
    cosine.operation = "DOT_PRODUCT"
    tree.links.new(geometry.outputs["Incoming"], cosine.inputs[0])
    tree.links.new(normalized_axis.outputs["Vector"], cosine.inputs[1])
    absolute_cosine = tree.nodes.new("ShaderNodeMath")
    absolute_cosine.name = "Absolute Distant Ray Cosine"
    absolute_cosine.operation = "ABSOLUTE"
    tree.links.new(cosine.outputs["Value"], absolute_cosine.inputs[0])
    profile = tree.nodes.new("ShaderNodeMath")
    profile.name = "Distant Radiance Profile"
    profile.operation = "POWER"
    profile.inputs[1].default_value = -3.0
    tree.links.new(absolute_cosine.outputs[0], profile.inputs[0])
    scaled_strength = tree.nodes.new("ShaderNodeMath")
    scaled_strength.name = "Distant Light Strength"
    scaled_strength.operation = "MULTIPLY"
    scaled_strength.inputs[0].default_value = strength / (1.0 + size * size)
    tree.links.new(profile.outputs[0], scaled_strength.inputs[1])
    tree.links.new(scaled_strength.outputs[0], emission.inputs["Strength"])


def configure_world(args: RenderOptions, yaws: list[float]) -> bpy.types.World:
    """Create the scene world graph."""

    world = bpy.data.worlds.new("BrickNet World")
    world[WORLD_OWNER_PROP] = RENDER_SCENE_NAME
    tree = world.node_tree
    bg = tree.nodes["Background"]
    bg.name = "Environment Background"
    source = environment_source(tree, args, yaws)
    if isinstance(source, tuple):
        bg.inputs["Color"].default_value = source
    else:
        tree.links.new(source, bg.inputs["Color"])
    output = next(node for node in tree.nodes if node.bl_idname == "ShaderNodeOutputWorld")
    for link in list(output.inputs["Surface"].links):
        tree.links.remove(link)

    def environment_radiance_factor(light_path):
        bounce = tree.nodes.new("ShaderNodeMath")
        bounce.name = "Diffuse or Glossy Ray"
        bounce.operation = "ADD"
        tree.links.new(light_path.outputs["Is Diffuse Ray"], bounce.inputs[0])
        tree.links.new(light_path.outputs["Is Glossy Ray"], bounce.inputs[1])
        before_transmission = tree.nodes.new("ShaderNodeMath")
        before_transmission.name = "Before Transmission"
        before_transmission.operation = "LESS_THAN"
        before_transmission.inputs[1].default_value = 0.5
        tree.links.new(light_path.outputs["Transmission Depth"], before_transmission.inputs[0])
        bounced = tree.nodes.new("ShaderNodeMath")
        bounced.name = "Bounced Environment Ray"
        bounced.operation = "MULTIPLY"
        tree.links.new(bounce.outputs["Value"], bounced.inputs[0])
        tree.links.new(before_transmission.outputs["Value"], bounced.inputs[1])
        factor = tree.nodes.new("ShaderNodeMath")
        factor.name = "Environment Radiance Factor"
        factor.operation = "ADD"
        factor.inputs[0].default_value = 1.0
        tree.links.new(bounced.outputs["Value"], factor.inputs[1])
        return factor.outputs["Value"]

    def scale_environment_strength(name, strength, factor):
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = name
        scale.operation = "MULTIPLY"
        scale.inputs[0].default_value = strength
        tree.links.new(factor, scale.inputs[1])
        return scale.outputs["Value"]

    def strength_by_ray_type(name, ordinary, unscaled):
        strength = tree.nodes.new("ShaderNodeMix")
        strength.name = name
        strength.data_type = "FLOAT"
        tree.links.new(unscaled, strength.inputs["Factor"])
        if isinstance(ordinary, (int, float)):
            strength.inputs["A"].default_value = ordinary
        else:
            tree.links.new(ordinary, strength.inputs["A"])
        strength.inputs["B"].default_value = 1.0
        return strength.outputs["Result"]

    def pure_refraction_path(light_path):
        glossy_depth = tree.nodes.new("ShaderNodeMath")
        glossy_depth.name = "No Glossy Bounce"
        glossy_depth.operation = "LESS_THAN"
        glossy_depth.inputs[1].default_value = 0.5
        tree.links.new(light_path.outputs["Glossy Depth"], glossy_depth.inputs[0])
        diffuse_depth = tree.nodes.new("ShaderNodeMath")
        diffuse_depth.name = "No Diffuse Bounce"
        diffuse_depth.operation = "LESS_THAN"
        diffuse_depth.inputs[1].default_value = 0.5
        tree.links.new(light_path.outputs["Diffuse Depth"], diffuse_depth.inputs[0])
        pure_refraction = tree.nodes.new("ShaderNodeMath")
        pure_refraction.name = "Pure Refraction Path"
        pure_refraction.operation = "MULTIPLY"
        tree.links.new(glossy_depth.outputs["Value"], pure_refraction.inputs[0])
        tree.links.new(diffuse_depth.outputs["Value"], pure_refraction.inputs[1])
        transmission = tree.nodes.new("ShaderNodeMath")
        transmission.name = "Pure Transmission Ray"
        transmission.operation = "MULTIPLY"
        tree.links.new(light_path.outputs["Is Transmission Ray"], transmission.inputs[0])
        tree.links.new(pure_refraction.outputs["Value"], transmission.inputs[1])
        return transmission.outputs["Value"]

    def camera_or_pure_refraction(light_path):
        result = tree.nodes.new("ShaderNodeMath")
        result.name = "Camera or Pure Refraction"
        result.operation = "MAXIMUM"
        tree.links.new(light_path.outputs["Is Camera Ray"], result.inputs[0])
        tree.links.new(pure_refraction_path(light_path), result.inputs[1])
        return result.outputs["Value"]

    if args.env_map is not None:
        light_path = tree.nodes.new("ShaderNodeLightPath")
        light_path.name = "Ray Type"
        pure_refraction = pure_refraction_path(light_path)
        radiance_factor = environment_radiance_factor(light_path)
        tree.links.new(
            strength_by_ray_type(
                "Environment Map Strength",
                scale_environment_strength("Environment Map Radiance", args.env_intensity, radiance_factor),
                pure_refraction,
            ),
            bg.inputs["Strength"],
        )
        lighting = bg.outputs[0]
        fill = tree.nodes.new("ShaderNodeBackground")
        fill.name = "Environment Fill Light"
        fill.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        tree.links.new(
            strength_by_ray_type(
                "Environment Fill Strength",
                scale_environment_strength("Environment Fill Radiance", args.env_effect, radiance_factor),
                pure_refraction,
            ),
            fill.inputs["Strength"],
        )
        add = tree.nodes.new("ShaderNodeAddShader")
        add.name = "Environment Map and Fill Light"
        tree.links.new(lighting, add.inputs[0])
        tree.links.new(fill.outputs[0], add.inputs[1])
        lighting = add.outputs[0]
        camera_bg = tree.nodes.new("ShaderNodeBackground")
        camera_bg.name = "Camera Background"
        camera_bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        camera_bg.inputs["Strength"].default_value = 2.0
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.name = "Camera and Environment Background"
        tree.links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
        tree.links.new(lighting, mix.inputs[1])
        tree.links.new(camera_bg.outputs[0], mix.inputs[2])
        tree.links.new(mix.outputs[0], output.inputs["Surface"])
        scene = bpy.context.scene
        scene.world = world
        scene.render.film_transparent = args.transparent
        scene.cycles.film_transparent_glass = args.transparent
        scene.cycles.film_transparent_roughness = 0.3
        world.cycles.sampling_method = "AUTOMATIC" if args.env_intensity > 0.0 else "NONE"
        return world

    if args.background_mode == "FLOOR" and args.light_preset in LIGHT_RIG_PRESET_NAMES:
        light_path = tree.nodes.new("ShaderNodeLightPath")
        light_path.name = "Ray Type"
        bg.name = "Floor Environment Light"
        bg.inputs["Color"].default_value = (*args.ground_color, 1.0)
        tree.links.new(
            scale_environment_strength(
                "Floor Environment Strength",
                args.env_effect,
                environment_radiance_factor(light_path),
            ),
            bg.inputs["Strength"],
        )
        visible_bg = tree.nodes.new("ShaderNodeBackground")
        visible_bg.name = "Floor Background"
        visible_bg.inputs["Color"].default_value = (*args.ground_color, 1.0)
        visible_bg.inputs["Strength"].default_value = 1.0
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.name = "Floor Background and Lighting"
        tree.links.new(camera_or_pure_refraction(light_path), mix.inputs["Fac"])
        tree.links.new(bg.outputs[0], mix.inputs[1])
        tree.links.new(visible_bg.outputs[0], mix.inputs[2])
        tree.links.new(mix.outputs[0], output.inputs["Surface"])
        scene = bpy.context.scene
        scene.world = world
        scene.render.film_transparent = False
        scene.cycles.film_transparent_glass = False
        scene.cycles.film_transparent_roughness = 0.3
        world.cycles.sampling_method = "AUTOMATIC"
        return world

    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.name = "Ray Type"
    tree.links.new(
        scale_environment_strength(
            "Environment Strength",
            args.env_effect,
            environment_radiance_factor(light_path),
        ),
        bg.inputs["Strength"],
    )

    camera_bg = tree.nodes.new("ShaderNodeBackground")
    camera_bg.name = "Camera and Refraction Background"
    camera_source = source
    if isinstance(camera_source, tuple):
        camera_bg.inputs["Color"].default_value = camera_source
    else:
        tree.links.new(camera_source, camera_bg.inputs["Color"])
    lighting = bg.outputs[0]
    ambient = ambient_light_radiance(args)
    if any(ambient):
        ambient_bg = tree.nodes.new("ShaderNodeBackground")
        ambient_bg.name = "Ambient Light"
        ambient_bg.inputs["Color"].default_value = (*ambient, 1.0)
        ambient_bg.inputs["Strength"].default_value = 1.0
        add_ambient = tree.nodes.new("ShaderNodeAddShader")
        add_ambient.name = "Environment and Ambient Light"
        tree.links.new(lighting, add_ambient.inputs[0])
        tree.links.new(ambient_bg.outputs[0], add_ambient.inputs[1])
        lighting = add_ambient.outputs[0]
    camera_background = camera_bg.outputs[0]
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "Background by Ray Type"
    tree.links.new(camera_or_pure_refraction(light_path), mix.inputs["Fac"])
    tree.links.new(lighting, mix.inputs[1])
    tree.links.new(camera_background, mix.inputs[2])
    tree.links.new(mix.outputs[0], output.inputs["Surface"])
    scene = bpy.context.scene
    scene.world = world
    scene.render.film_transparent = args.transparent
    scene.cycles.film_transparent_glass = args.transparent
    scene.cycles.film_transparent_roughness = 0.3
    world.cycles.sampling_method = (
        "AUTOMATIC" if args.env_map is not None and args.env_intensity > 0.0 and args.env_effect > 0.0 else "NONE"
    )
    return world


def add_lights(args: RenderOptions, yaws: list[float]) -> bpy.types.Object | None:
    """Create the selected distant-light rig."""

    specs = LIGHT_PRESETS[args.light_preset]
    if not specs or args.light_intensity <= 0.0:
        return None

    color = _resolved_light_color(args)
    rig = bpy.data.objects.new("Lighting (%s)" % args.light_preset, None)
    rig.empty_display_type = "PLAIN_AXES"
    rig.empty_display_size = 20.0
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = resolved_light_rotation(args, yaws[0])
    link_object(rig)

    for spec in (spec for spec in specs if spec.type == "distant"):
        data = bpy.data.lights.new(name=spec.name, type="SUN")
        data.energy = 1.0
        data.angle = distant_size_to_sun_angle(spec.size)
        configure_distant_light_shader(
            data,
            tuple(spec.color[c] * color[c] for c in range(3)),
            spec.strength * args.light_intensity,
            spec.size,
        )
        obj = bpy.data.objects.new(name=spec.name, object_data=data)
        obj.parent = rig
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = Matrix(spec.matrix)
        link_object(obj)

    if len(yaws) > 1:
        scene = bpy.context.scene
        for frame, yaw in enumerate(yaws):
            rig.rotation_euler = resolved_light_rotation(args, yaw)
            rig.keyframe_insert(data_path="rotation_euler", frame=frame)
        scene.frame_set(scene.frame_start)
    return rig


def select_compute_devices(preferences, requested: str):
    """Return the selected Cycles compute type and matching GPU devices."""

    requested = (str(requested).strip() or "AUTO").upper()
    candidates = GPU_COMPUTE_ORDER if requested == "AUTO" else (requested,)
    for compute in candidates:
        try:
            preferences.compute_device_type = compute
            preferences.refresh_devices()
        except TypeError as exc:
            if requested == "AUTO":
                continue
            raise RuntimeError("unsupported GPU compute backend %s" % requested) from exc
        devices = [device for device in preferences.devices if device.type == compute]
        if devices:
            return compute, devices
        if requested != "AUTO":
            raise RuntimeError("GPU compute backend %s has no available devices" % requested)
    raise RuntimeError("no supported GPU compute devices found; tried %s" % ", ".join(candidates))


def configure_render(args: RenderOptions) -> None:
    """Apply Cycles, viewport, resolution, and color-management settings."""

    global CYCLES_DEVICE_CONFIG

    scene = bpy.context.scene
    res_x, res_y = (int(value) for value in args.resolution.split("x"))
    scene.render.engine = "CYCLES"
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.denoising_prefilter = "ACCURATE"
    scene.cycles.denoising_quality = "HIGH"
    scene.cycles.max_bounces = 32
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 32
    scene.cycles.volume_bounces = 32
    scene.cycles.sample_clamp_direct = 1131.0
    scene.cycles.sample_clamp_indirect = 40.0
    scene.cycles.light_sampling_threshold = 0.01
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.use_animated_seed = True
    scene.cycles.seed = args.random_seed
    scene.cycles.use_light_tree = False
    scene.display.shading.color_type = "MATERIAL"
    if args.device == "CPU":
        scene.cycles.device = "CPU"
        scene.cycles.denoising_use_gpu = False
        CYCLES_DEVICE_CONFIG = ("CPU", "")
    else:
        scene.cycles.device = "GPU"
        scene.cycles.denoising_use_gpu = True
        requested_compute = args.compute
        key = ("GPU", requested_compute)
        if CYCLES_DEVICE_CONFIG != key:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            selected_compute, _selected_devices = select_compute_devices(prefs, requested_compute)
            for dev in prefs.devices:
                dev.use = dev.type == selected_compute
            CYCLES_DEVICE_CONFIG = key
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = args.adaptive_threshold > 0.0
    scene.cycles.adaptive_threshold = args.adaptive_threshold
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.filter_size = 1.5
    for prop in scene.render.bl_rna.properties:
        if prop.identifier.startswith("use_stamp_"):
            setattr(scene.render, prop.identifier, False)
    scene.render.use_motion_blur = False
    scene.render.use_persistent_data = True
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.view_settings.view_transform = "Standard"


def owned_view_layer(
    scene: bpy.types.Scene,
    name: str,
    owner: str,
    *,
    reuse_first: bool = False,
) -> bpy.types.ViewLayer:
    """Return a view layer after the caller has checked name ownership."""

    existing = scene.view_layers.get(name)
    if existing is not None:
        return existing
    layer = scene.view_layers[0] if reuse_first else None
    if layer is None or layer.get(VIEW_LAYER_OWNER_PROP) not in {None, owner}:
        layer = scene.view_layers.new(name)
    else:
        layer.name = name
    layer[VIEW_LAYER_OWNER_PROP] = owner
    return layer


def remove_owned_compositors(scene: bpy.types.Scene) -> None:
    """Remove compositor groups created by this package."""

    if (
        scene.compositing_node_group is not None
        and scene.compositing_node_group.get(COMPOSITOR_OWNER_PROP) == COMPOSITOR_NAME
    ):
        scene.compositing_node_group = None
    for group in tuple(bpy.data.node_groups):
        if group.get(COMPOSITOR_OWNER_PROP) == COMPOSITOR_NAME:
            bpy.data.node_groups.remove(group)


def _ground_direct_irradiance(light_rig: bpy.types.Object | None) -> float:
    """Return unoccluded direct irradiance on the horizontal ground."""

    if light_rig is None:
        return 0.0
    samples = 32
    irradiance = np.zeros(3)
    for obj in light_rig.children:
        light = obj.data
        if not isinstance(light, bpy.types.SunLight):
            continue
        axis = light_rig.matrix_basis.to_3x3() @ obj.matrix_basis.to_3x3() @ Vector((0.0, 0.0, 1.0))
        z = axis.normalized().z
        size = math.tan(0.5 * light.angle)
        cosine = 1.0 / math.sqrt(1.0 + size * size)
        mu = cosine + (np.arange(samples) + 0.5) * ((1.0 - cosine) / samples)
        a = z * mu
        b = math.sqrt(max(0.0, 1.0 - z * z)) * np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
        ratio = np.divide(a, b, out=np.ones_like(a), where=b > 0.0)
        positive = np.where(
            a >= b,
            a,
            (a * np.arccos(np.clip(-ratio, -1.0, 1.0)) + b * np.sqrt(np.maximum(0.0, 1.0 - ratio * ratio))) / math.pi,
        )
        emission = light.node_tree.nodes["Emission"]
        strength = emission.inputs["Strength"].default_value
        if size > 0.0:
            strength = light.node_tree.nodes["Distant Light Strength"].inputs[0].default_value
        irradiance += (
            np.asarray(emission.inputs["Color"].default_value[:3])
            * strength
            * (2.0 / (math.pi * (1.0 + cosine)))
            * np.mean(positive * mu**-3)
        )
    return float(np.mean(irradiance))


def configure_compositor(args: RenderOptions, light_rig: bpy.types.Object | None = None) -> bool:
    """Build the requested compositor graph. Returns True when active."""

    scene = bpy.context.scene
    mask_size = projection_mask_size(args)
    shadow_ground = args.ground_mode == "SHADOW"
    has_shadow_catcher = any(
        obj.type == "MESH" and obj.is_shadow_catcher and not obj.hide_render for obj in scene.objects
    )
    if not args.postprocess and not args.alpha_adjust and mask_size is None and not shadow_ground:
        scene.render.use_compositing = False
        scene.compositing_node_group = None
        remove_owned_compositors(scene)
        layer = scene.view_layers.get(MAIN_VIEW_LAYER)
        layer = layer if layer is not None and layer.get(VIEW_LAYER_OWNER_PROP) == MAIN_VIEW_LAYER_OWNER else None
        layer = layer or scene.view_layers[0]
        for name in ("ambient_occlusion", "material_index", "z"):
            setattr(layer, "use_pass_" + name, False)
        layer.use_pass_emit = False
        layer.use_pass_environment = False
        layer.cycles.use_pass_shadow_catcher = shadow_ground
        return False
    existing_layer = scene.view_layers.get(MAIN_VIEW_LAYER)
    if existing_layer is not None and existing_layer.get(VIEW_LAYER_OWNER_PROP) != MAIN_VIEW_LAYER_OWNER:
        raise RuntimeError("view-layer name %s is already in use" % MAIN_VIEW_LAYER)
    existing_tree = bpy.data.node_groups.get(COMPOSITOR_NAME)
    if existing_tree is not None and existing_tree.get(COMPOSITOR_OWNER_PROP) != COMPOSITOR_NAME:
        raise RuntimeError("node-group name %s is already in use" % COMPOSITOR_NAME)
    remove_owned_compositors(scene)
    scene.render.use_compositing = True
    try:
        import gpu

        gpu.platform.vendor_get()
        scene.render.compositor_device = "GPU"
    except (ImportError, RuntimeError, SystemError, TypeError):
        scene.render.compositor_device = "CPU"
    layer = owned_view_layer(scene, MAIN_VIEW_LAYER, MAIN_VIEW_LAYER_OWNER, reuse_first=True)
    layer.use = True
    floor_shadow = shadow_ground and args.background_mode == "FLOOR"
    direct = _ground_direct_irradiance(light_rig) if floor_shadow else 0.0
    environment = args.env_effect * float(np.mean(args.ground_color))
    denominator = direct + environment
    correct_floor_shadow = floor_shadow and denominator > 0.0 and 2.0 * environment >= direct
    layer.use_pass_ambient_occlusion = args.postprocess or correct_floor_shadow
    for name in ("material_index", "z"):
        setattr(layer, "use_pass_" + name, args.postprocess)
    layer.use_pass_emit = False
    layer.use_pass_environment = shadow_ground and not args.transparent
    layer.cycles.use_pass_shadow_catcher = shadow_ground

    tree = bpy.data.node_groups.new(COMPOSITOR_NAME, "CompositorNodeTree")
    tree[COMPOSITOR_OWNER_PROP] = COMPOSITOR_NAME
    scene.compositing_node_group = tree
    links = tree.links

    def link_or_set(socket, value):
        if isinstance(value, (int, float, tuple)):
            socket.default_value = value
        else:
            links.new(value, socket)

    def scalar_math(name, op, left, right):
        node = tree.nodes.new("ShaderNodeMath")
        node.name = name
        node.operation = op
        link_or_set(node.inputs[0], left)
        link_or_set(node.inputs[1], right)
        return node.outputs[0]

    def split_color(name, source):
        node = tree.nodes.new("CompositorNodeSeparateColor")
        node.name = name
        links.new(source, node.inputs["Image"])
        return node.outputs

    def reconstruct_shadow_catcher(source, shadow, environment=None):
        image_channels = split_color("Shadow Matte", source)
        shadow_channels = split_color("Shadow Ratio", shadow)
        transparency = scalar_math("Shadow Matte Transparency", "SUBTRACT", 1.0, image_channels["Alpha"])
        if environment is None:
            weighted = [
                scalar_math("Shadow %s Luminance" % c, "MULTIPLY", shadow_channels[c], weight)
                for c, weight in zip(("Red", "Green", "Blue"), (0.2126, 0.7152, 0.0722))
            ]
            luminance = scalar_math(
                "Shadow Luminance",
                "ADD",
                scalar_math("Shadow Red + Green", "ADD", weighted[0], weighted[1]),
                weighted[2],
            )
            alpha = scalar_math(
                "Transparent Shadow Alpha",
                "SUBTRACT",
                1.0,
                scalar_math("Shadow Transparency", "MULTIPLY", transparency, luminance),
            )
            result = tree.nodes.new("CompositorNodeSetAlpha")
            result.name = "Transparent Shadow Composite"
            result.inputs["Type"].default_value = "Replace Alpha"
            links.new(source, result.inputs["Image"])
            links.new(alpha, result.inputs["Alpha"])
            return result.outputs["Image"]

        environment_channels = split_color("Shadow Background", environment)
        shadow_sum = scalar_math(
            "Shadow RGB Sum",
            "ADD",
            scalar_math("Shadow Red + Green", "ADD", shadow_channels["Red"], shadow_channels["Green"]),
            shadow_channels["Blue"],
        )
        shadow_ratio = scalar_math("Neutral Shadow Ratio", "DIVIDE", shadow_sum, 3.0)
        result = tree.nodes.new("CompositorNodeCombineColor")
        result.name = "Opaque Shadow Composite"
        for c in ("Red", "Green", "Blue"):
            background = scalar_math(
                "Shadowed Background " + c,
                "MULTIPLY",
                environment_channels[c],
                shadow_ratio,
            )
            if not has_shadow_catcher:
                background = scalar_math(
                    "Visible Background " + c,
                    "MULTIPLY",
                    background,
                    transparency,
                )
            links.new(
                scalar_math("Shadow Composite " + c, "ADD", image_channels[c], background),
                result.inputs[c],
            )
        result.inputs["Alpha"].default_value = 1.0
        return result.outputs["Image"]

    def adjust_alpha(source):
        channels = tree.nodes.new("CompositorNodeSeparateColor")
        channels.name = "Alpha Adjust Input"
        links.new(source, channels.inputs["Image"])
        inverse_alpha = scalar_math("One Minus Alpha", "SUBTRACT", 1.0, channels.outputs["Alpha"])
        lifted = tree.nodes.new("CompositorNodeCombineColor")
        lifted.name = "RGB Plus Transparency"
        for c in ("Red", "Green", "Blue"):
            links.new(
                scalar_math("Lifted " + c, "ADD", channels.outputs[c], inverse_alpha),
                lifted.inputs[c],
            )
        lifted.inputs["Alpha"].default_value = 1.0

        srgb = tree.nodes.new("CompositorNodeConvertColorSpace")
        srgb.name = "Input in sRGB"
        srgb.from_color_space = "scene_linear"
        srgb.to_color_space = "sRGB"
        links.new(source, srgb.inputs["Image"])
        lifted_srgb = tree.nodes.new("CompositorNodeConvertColorSpace")
        lifted_srgb.name = "Lifted RGB in sRGB"
        lifted_srgb.from_color_space = "scene_linear"
        lifted_srgb.to_color_space = "sRGB"
        links.new(lifted.outputs["Image"], lifted_srgb.inputs["Image"])
        srgb_channels = tree.nodes.new("CompositorNodeSeparateColor")
        lifted_channels = tree.nodes.new("CompositorNodeSeparateColor")
        links.new(srgb.outputs["Image"], srgb_channels.inputs["Image"])
        links.new(lifted_srgb.outputs["Image"], lifted_channels.inputs["Image"])
        weighted_delta = [
            scalar_math(
                "Alpha Adjust %s Delta" % c,
                "MULTIPLY",
                scalar_math(
                    "Alpha Adjust %s Difference" % c,
                    "SUBTRACT",
                    lifted_channels.outputs[c],
                    srgb_channels.outputs[c],
                ),
                weight,
            )
            for c, weight in zip(("Red", "Green", "Blue"), (0.30, 0.59, 0.11))
        ]
        delta = scalar_math(
            "Alpha Adjust Weighted Delta",
            "ADD",
            scalar_math("Alpha Adjust Red + Green", "ADD", weighted_delta[0], weighted_delta[1]),
            weighted_delta[2],
        )
        luminance = scalar_math(
            "Alpha Adjust Luminance",
            "MAXIMUM",
            0.0,
            scalar_math(
                "Clamped Alpha Adjust Luminance", "MINIMUM", 1.0, scalar_math("One Minus Delta", "SUBTRACT", 1.0, delta)
            ),
        )
        set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
        set_alpha.name = "Alpha Adjustment Output"
        set_alpha.inputs["Type"].default_value = "Replace Alpha"
        links.new(source, set_alpha.inputs["Image"])
        alpha = channels.outputs["Alpha"]
        adjusted = scalar_math("Adjusted Alpha", "MAXIMUM", alpha, luminance)
        intermediate = scalar_math(
            "Intermediate Alpha",
            "MULTIPLY",
            scalar_math("Alpha Above Zero", "GREATER_THAN", alpha, 0.0),
            scalar_math("Alpha Below One", "LESS_THAN", alpha, 1.0),
        )
        links.new(
            scalar_math(
                "Gated Alpha Adjust",
                "ADD",
                alpha,
                scalar_math(
                    "Eligible Alpha Delta",
                    "MULTIPLY",
                    intermediate,
                    scalar_math("Alpha Adjust Delta", "SUBTRACT", adjusted, alpha),
                ),
            ),
            set_alpha.inputs["Alpha"],
        )
        return set_alpha.outputs["Image"]

    def mask_projection(source):
        if mask_size is None:
            return source
        ellipse = tree.nodes.new("CompositorNodeEllipseMask")
        ellipse.name = "Valid Camera Projection"
        ellipse.inputs["Position"].default_value = (0.5, 0.5)
        ellipse.inputs["Size"].default_value = mask_size
        channels = tree.nodes.new("CompositorNodeSeparateColor")
        channels.name = "Projection Mask Input"
        links.new(source, channels.inputs["Image"])
        set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
        set_alpha.name = "Projection Alpha"
        set_alpha.inputs["Type"].default_value = "Replace Alpha"
        links.new(source, set_alpha.inputs["Image"])
        links.new(
            scalar_math(
                "Masked Projection Alpha",
                "MULTIPLY",
                channels.outputs["Alpha"],
                ellipse.outputs["Mask"],
            ),
            set_alpha.inputs["Alpha"],
        )
        return set_alpha.outputs["Image"]

    def output_image(source):
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        out = tree.nodes.new("NodeGroupOutput")
        out.name = "Composite Output"
        links.new(mask_projection(source), out.inputs[0])

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.name = "Main Render Layers"
    render_layers.scene = scene
    render_layers.layer = layer.name
    image = render_layers.outputs["Image"]
    if shadow_ground:
        shadow = render_layers.outputs["Shadow Catcher"]
        if correct_floor_shadow:
            shadow_channels = split_color("Input Shadow Ratio", shadow)
            ao_channels = split_color("Environment Occlusion", render_layers.outputs["Ambient Occlusion"])
            shadow_sum = scalar_math(
                "Input Shadow Sum",
                "ADD",
                scalar_math("Input Shadow Red + Green", "ADD", shadow_channels["Red"], shadow_channels["Green"]),
                shadow_channels["Blue"],
            )
            shadow_ratio = scalar_math("Neutral Input Shadow", "DIVIDE", shadow_sum, 3.0)
            corrected = tree.nodes.new("CompositorNodeCombineColor")
            corrected.name = "Corrected Shadow Ratio RGB"
            ratio = scalar_math(
                "Corrected Shadow Ratio",
                "DIVIDE",
                scalar_math(
                    "Corrected Shadow Numerator",
                    "SUBTRACT",
                    scalar_math(
                        "Weighted Input Shadow",
                        "MULTIPLY",
                        shadow_ratio,
                        direct + 2.0 * environment,
                    ),
                    scalar_math(
                        "Environment Shadow Contribution",
                        "MULTIPLY",
                        ao_channels["Red"],
                        environment,
                    ),
                ),
                denominator,
            )
            for c in ("Red", "Green", "Blue"):
                links.new(ratio, corrected.inputs[c])
            corrected.inputs["Alpha"].default_value = 1.0
            shadow = corrected.outputs["Image"]
        image = reconstruct_shadow_catcher(
            image,
            shadow,
            None if args.transparent else render_layers.outputs["Environment"],
        )
    if not args.postprocess:
        if args.alpha_adjust:
            image = adjust_alpha(image)
        output_image(image)
        return True
    ao_source = render_layers.outputs["Ambient Occlusion"]
    material_index = render_layers.outputs["Material Index"]
    depth = render_layers.outputs["Depth"]

    def material_mask(name: str, index: int):
        # ID masks encode discrete material classes; filtering their edges would
        # mix AO policies between classes.
        mask = tree.nodes.new("CompositorNodeIDMask")
        mask.name = name
        mask.inputs["Index"].default_value = index
        links.new(material_index, mask.inputs["ID value"])
        return mask.outputs["Alpha"]

    image_channels = tree.nodes.new("CompositorNodeSeparateColor")
    image_channels.name = "Main Image Channels"
    links.new(image, image_channels.inputs["Image"])
    image_alpha = image_channels.outputs["Alpha"]
    transmissive_alpha = scalar_math("Transmissive Alpha", "MINIMUM", image_alpha, 1.0 - TRANSMISSIVE_AO_RESIDUAL)
    transmissive_exempt_mask = scalar_math(
        "Transmissive Material Mask",
        "MAXIMUM",
        material_mask("Transmissive AO Material Mask", 2),
        material_mask("Transmissive Local AO Material Mask", 4),
    )
    transmissive_exempt_mask = scalar_math(
        "Visible Transmissive Mask", "MULTIPLY", transmissive_exempt_mask, transmissive_alpha
    )
    exempt_mask = scalar_math(
        "AO Exemption Mask", "MAXIMUM", transmissive_exempt_mask, material_mask("Local AO Material Mask", 3)
    )
    ao_minus_one = scalar_math("AO Minus One", "SUBTRACT", ao_source, 1.0)
    exempt_delta = scalar_math("Exempt AO Delta", "MULTIPLY", ao_minus_one, exempt_mask)
    ao_eff = scalar_math("Material-Adjusted AO", "SUBTRACT", ao_source, exempt_delta)
    sky_mask = scalar_math("Sky Mask", "GREATER_THAN", depth, 1.0e7)
    sky_delta = scalar_math("Sky AO Delta", "MULTIPLY", ao_minus_one, sky_mask)
    ao_eff = scalar_math("Effective AO", "SUBTRACT", ao_eff, sky_delta)

    def ao_scalar_factor(value):
        contrast = scalar_math("AO Contrast", "MULTIPLY", value, 0.75)
        return scalar_math("AO Factor", "ADD", contrast, 0.25)

    def scalar_to_color(value):
        color = tree.nodes.new("CompositorNodeCombineColor")
        color.name = "AO Factor Color"
        for name in ("Red", "Green", "Blue"):
            links.new(value, color.inputs[name])
        color.inputs["Alpha"].default_value = 1.0
        return color.outputs["Image"]

    def multiply_image(left, right):
        node = tree.nodes.new("ShaderNodeMix")
        node.name = "Apply AO"
        node.data_type = "RGBA"
        node.blend_type = "MULTIPLY"
        node.clamp_result = False
        node.inputs["Factor"].default_value = 1.0
        links.new(left, node.inputs["A"])
        links.new(right, node.inputs["B"])
        return node.outputs["Result"]

    image = multiply_image(image, scalar_to_color(ao_scalar_factor(ao_eff)))
    if args.alpha_adjust:
        image = adjust_alpha(image)
    output_image(image)
    return True


def save_blend(args: RenderOptions) -> None:
    """Write a .blend file when requested."""

    if args.save_blend:
        for node_group in {
            modifier.node_group
            for obj in bpy.data.objects
            for modifier in obj.modifiers
            if modifier.type == "NODES" and modifier.node_group is not None and modifier.node_group.library is not None
        }:
            node_group.make_local()
        bpy.ops.file.pack_all()
        path = Path(args.save_blend).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path))


def render_output_path(args: RenderOptions, stem: str, frame: int) -> Path:
    """Return the output image path for one frame."""

    tag = "%s_%04d" % (stem, frame) if args.views > 1 else stem
    extension = "png" if args.format in {"png8", "png16"} else args.format
    return Path(args.out_dir) / ("%s.%s" % (tag, extension))


def configure_image_format(args: RenderOptions) -> str:
    """Apply render image settings and return the output file extension."""

    scene = bpy.context.scene
    scene.render.image_settings.media_type = "IMAGE"
    image_format = args.format
    if image_format in {"png8", "png16"}:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8" if image_format == "png8" else "16"
        scene.render.image_settings.compression = PNG_COMPRESSION
        return "png"
    if image_format == "jpg":
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.image_settings.quality = JPEG_QUALITY
        return "jpg"
    if image_format == "bmp":
        scene.render.image_settings.file_format = "BMP"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        return "bmp"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = (
        "RGBA" if args.transparent or projection_mask_size(args) is not None else "RGB"
    )
    scene.render.image_settings.color_depth = "32"
    return "exr"


def turntable_output_paths(args: RenderOptions) -> list[Path]:
    """Return the files produced by the configured turntable."""

    turntable = args.turntable
    if turntable is None:
        raise ValueError("turntable configuration is required")
    prefix = Path(args.out_dir) / (Path(args.ldr_path).stem + "_")
    if turntable.format in {"png8", "png16"}:
        return [Path("%s%04d.png" % (prefix, frame)) for frame in range(turntable.frame_end + 1)]
    return [Path("%s%04d-%04d.%s" % (prefix, 0, turntable.frame_end, turntable.format))]


def encode_gif(source: Path, destination: Path, ffmpeg: str | None = None) -> None:
    """Encode a lossless animation intermediate as a palette GIF."""

    executable = ffmpeg or shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("GIF output requires an ffmpeg executable on PATH or --ffmpeg")
    subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            "[0:v] split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1",
            "-f",
            "gif",
            str(destination),
        ],
        check=True,
    )
    source.unlink()


def configure_turntable_format(
    args: RenderOptions,
    scene: bpy.types.Scene | None = None,
) -> None:
    """Apply Blender's animation or lossless GIF-intermediate settings."""

    turntable = args.turntable
    if turntable is None:
        raise ValueError("turntable configuration is required")
    scene = scene or bpy.context.scene
    settings = scene.render.image_settings
    if turntable.format in {"png8", "png16"}:
        settings.media_type = "IMAGE"
        settings.file_format = "PNG"
        settings.color_mode = "RGBA"
        settings.color_depth = "8" if turntable.format == "png8" else "16"
        settings.compression = PNG_COMPRESSION
        return
    settings.media_type = "VIDEO"
    settings.file_format = "FFMPEG"
    settings.color_mode = "RGB"
    settings.color_depth = "8"
    if turntable.format == "gif":
        scene.render.ffmpeg.format = "AVI"
        scene.render.ffmpeg.codec = "FFV1"
    else:
        scene.render.ffmpeg.format = "MPEG4" if turntable.format == "mp4" else "QUICKTIME"
        scene.render.ffmpeg.codec = "MPEG4"


def configure_output(args: RenderOptions) -> None:
    """Apply the requested output format and path to the scene."""

    stem = Path(args.ldr_path).stem
    if args.turntable is not None:
        configure_turntable_format(args)
        bpy.context.scene.render.filepath = str(Path(args.out_dir) / (stem + "_"))
    else:
        configure_image_format(args)
        bpy.context.scene.render.filepath = str(
            Path(args.out_dir) / (stem + "_") if args.views > 1 else render_output_path(args, stem, 0)
        )


def _render_temporally_denoised_turntable(args: RenderOptions) -> None:
    """Render, temporally denoise, and composite one animation."""

    scene = bpy.context.scene
    layer = scene.view_layers.get(MAIN_VIEW_LAYER) or scene.view_layers[0]
    final_filepath = scene.render.filepath
    source_tree = scene.compositing_node_group if scene.render.use_compositing else None
    previous = (scene.render.use_compositing, layer.use_pass_vector, layer.cycles.denoising_store_passes)

    with ExitStack() as resources:
        temporary = Path(resources.enter_context(TemporaryDirectory(prefix=".bricknet-temporal-", dir=args.out_dir)))
        try:
            raw_prefix = temporary / "raw_"
            denoised_prefix = temporary / "denoised_"
            settings = scene.render.image_settings
            scene.render.use_compositing = False
            layer.use_pass_vector = True
            layer.cycles.denoising_store_passes = True
            settings.media_type = "MULTI_LAYER_IMAGE"
            settings.file_format = "OPEN_EXR_MULTILAYER"
            settings.color_mode = "RGBA"
            settings.color_depth = "32"
            settings.exr_codec = "ZIP"
            scene.render.filepath = str(denoised_prefix)
            frames = range(scene.frame_start, scene.frame_end + 1)
            denoised_paths = [Path(scene.render.frame_path(frame=frame)) for frame in frames]
            scene.render.filepath = str(raw_prefix)
            raw_paths = [Path(scene.render.frame_path(frame=frame)) for frame in frames]

            bpy.ops.render.render(scene=scene.name, animation=True)
            with bpy.context.temp_override(scene=scene, view_layer=layer):
                bpy.ops.cycles.denoise_animation(output_filepath=str(denoised_prefix))
            missing = [path for path in denoised_paths if not path.is_file() or path.stat().st_size == 0]
            if missing:
                raise RuntimeError("Cycles temporal denoising did not produce frames: %s" % missing)

            images = [
                bpy.data.images.load(str(path), check_existing=False) for path in (denoised_paths[0], raw_paths[0])
            ]
            for image in images:
                resources.callback(bpy.data.images.remove, image)
                image.source = "SEQUENCE"
            tree = (
                source_tree.copy()
                if source_tree is not None
                else bpy.data.node_groups.new(".BrickNet Temporal Composite", "CompositorNodeTree")
            )
            resources.callback(bpy.data.node_groups.remove, tree)
            image_nodes = [tree.nodes.new("CompositorNodeImage") for image in images]
            for image_node, image in zip(image_nodes, images):
                image_node.image = image
                image_node.layer = layer.name
                image_node.frame_duration = scene.frame_end - scene.frame_start + 1
                image_node.frame_start = scene.frame_start
                image_node.frame_offset = scene.frame_start - 1
                image_node.use_auto_refresh = True
            clean_node, raw_node = image_nodes
            if source_tree is None:
                tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
                tree.links.new(clean_node.outputs["Combined"], tree.nodes.new("NodeGroupOutput").inputs["Image"])
            else:
                render_layers = tree.nodes["Main Render Layers"]
                for link in tuple(link for socket in render_layers.outputs for link in socket.links):
                    target = link.to_socket
                    source_name = link.from_socket.name
                    tree.links.remove(link)
                    source = clean_node.outputs["Combined"] if source_name == "Image" else raw_node.outputs[source_name]
                    tree.links.new(source, target)
                tree.nodes.remove(render_layers)

            output_scene = bpy.data.scenes.new(".BrickNet Temporal Output")
            resources.callback(bpy.data.scenes.remove, output_scene)
            resources.callback(setattr, output_scene, "compositing_node_group", None)
            output_scene.collection.objects.link(scene.camera)
            output_scene.camera = scene.camera
            output_scene.render.engine = "CYCLES"
            output_scene.cycles.samples = 1
            output_scene.cycles.use_denoising = False
            output_scene.render.compositor_device = scene.render.compositor_device
            output_scene.frame_start = scene.frame_start
            output_scene.frame_end = scene.frame_end
            render_names = "resolution_x resolution_y resolution_percentage pixel_aspect_x pixel_aspect_y fps fps_base"
            for name in render_names.split():
                setattr(output_scene.render, name, getattr(scene.render, name))
            output_scene.display_settings.display_device = scene.display_settings.display_device
            for name in ("view_transform", "look", "exposure", "gamma"):
                setattr(output_scene.view_settings, name, getattr(scene.view_settings, name))
            configure_turntable_format(args, output_scene)
            output_scene.render.filepath = final_filepath
            output_scene.render.use_compositing = True
            output_scene.compositing_node_group = tree
            bpy.ops.render.render(scene=output_scene.name, animation=True)
        finally:
            scene.render.use_compositing, layer.use_pass_vector, layer.cycles.denoising_store_passes = previous
            configure_turntable_format(args, scene)
            scene.render.filepath = final_filepath


def render_turntable(args: RenderOptions) -> list[Path]:
    """Render a turntable with Blender's animation output."""

    gif_encoder = None
    if args.turntable is not None and args.turntable.format == "gif":
        gif_encoder = args.turntable.ffmpeg or shutil.which("ffmpeg")
        if gif_encoder is None:
            raise RuntimeError("GIF output requires an ffmpeg executable on PATH or --ffmpeg")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    configure_turntable_format(args)
    bpy.context.scene.render.filepath = str(Path(args.out_dir) / (Path(args.ldr_path).stem + "_"))
    if args.turntable is not None and args.turntable.temporal_denoise:
        _render_temporally_denoised_turntable(args)
    else:
        bpy.ops.render.render(animation=True)
    outputs = turntable_output_paths(args)
    if gif_encoder is not None:
        encode_gif(outputs[0].with_suffix(".avi"), outputs[0], gif_encoder)
    return outputs


def render_views(args: RenderOptions) -> list[Path]:
    """Render every requested view."""

    views = args.views
    scene = bpy.context.scene
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(args.ldr_path).stem
    outputs = [render_output_path(args, stem, frame) for frame in range(views)]

    configure_image_format(args)
    for frame, output in enumerate(outputs):
        scene.frame_set(frame)
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
    return outputs


def build_scene(
    args: RenderOptions,
    map_material: MaterialMapper | None = None,
) -> tuple[list[bpy.types.Object], np.ndarray, list[float]]:
    """Build one Blender scene and return mesh objects, Y-up points, and view yaws."""

    if args.material is not None and map_material is not None:
        raise ValueError("material and map_material are mutually exclusive")
    clear_scene(keep_part_templates=False)
    scene = bpy.context.scene
    builds = args.turntable is not None and args.turntable.build_steps
    mesh_objs, points = (
        load_model_objects(args, load_placements(args.ldr_path, args.glb_dir), map_material)
        if builds
        else load_model(args, map_material=map_material)
    )
    static_bounds = mesh_world_bounds(mesh_objs) if builds and args.turntable.falling else None
    yaws = (
        build_turntable(scene, args, mesh_objs, points)
        if args.turntable is not None
        else build_orbit(scene, args, mesh_objs, points)
    )
    build_materials(args, mesh_objs)
    add_ground(args, mesh_objs, bounds=static_bounds)
    lighting_yaws = [args.yaw0] if args.turntable is not None else yaws
    configure_world(args, lighting_yaws)
    light_rig = add_lights(args, lighting_yaws)
    configure_render(args)
    configure_compositor(args, light_rig)
    return mesh_objs, points, yaws


def render_ldr(
    ldr_path: str | Path,
    out_dir: str | Path,
    *,
    glb_dir: str | Path | None = None,
    map_material: MaterialMapper | None = None,
    **overrides,
) -> list[Path]:
    """Render an LDraw model and return its output image paths."""

    options = RenderOptions(
        ldr_path=str(ldr_path),
        out_dir=str(out_dir),
        glb_dir=str(default_glb_dir() if glb_dir is None else glb_dir),
        **overrides,
    )
    _mesh_objs, _points, _yaws = build_scene(options, map_material=map_material)
    configure_output(options)
    save_blend(options)
    if options.no_render:
        return []
    return render_turntable(options) if options.turntable is not None else render_views(options)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    args = parse_args(argv, prog=prog)
    _mesh_objs, _points, _yaws = build_scene(args)
    configure_output(args)
    save_blend(args)
    if args.no_render:
        return
    render_turntable(args) if args.turntable is not None else render_views(args)


if __name__ == "__main__":
    main()
