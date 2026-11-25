"""Define the material catalog and its parameterized builders."""

from collections.abc import Callable
from functools import partial

import bpy
from mathutils import Color

from . import curve_data
from . import node_groups as ng
from . import node_utils
from .node_utils import Nodes, NodeWrangler

NODE_GROUPS: dict[str, Callable[[], bpy.types.NodeTree]] = {
    "CHROME-ANTIQUE-GROUP": ng.nodegroup_chrome_antique_group,
    "CHROME-GROUP": ng.nodegroup_chrome_group,
    "GLITTER-TRANS-GROUP": ng.nodegroup_glitter_trans_group,
    "GLITTER-TRANS_NEON-GROUP": ng.nodegroup_glitter_trans_neon_group,
    "METAL-GROUP": ng.nodegroup_metal_group,
    "PEARL-FLAT-GROUP": ng.nodegroup_pearl_flat_group,
    "PEARL-GROUP": ng.nodegroup_pearl_group,
    "PRINCIPLED-GROUP": ng.nodegroup_principled_group,
    "RUBBER-GROUP": ng.nodegroup_rubber_group,
    "RUBBER-TRANS-GROUP": ng.nodegroup_rubber_trans_group,
    "SATIN-GROUP": ng.nodegroup_satin_group,
    "SATIN-TRANS-GROUP": ng.nodegroup_satin_trans_group,
    "SHINE_GROUP": ng.nodegroup_shine_group,
    "SOLID-GROUP": ng.nodegroup_solid_group,
    "SPECKLE-GROUP": ng.nodegroup_speckle_group,
    "TRANS-GROUP_BASE": ng.nodegroup_trans_group_base,
    "TRANS-NEON-GROUP": ng.nodegroup_trans_neon_group,
}

# LDraw color codes supported by the renderer. Context-dependent codes
# 16 and 24 use white and black for top-level placements; codes 494 and 495
# share the silver and copper catalog materials.
LDRAW_MATERIAL_NAMES: dict[int, str] = {
    0: "SOLID-BLACK",
    1: "SOLID-BLUE",
    2: "SOLID-GREEN",
    3: "SOLID-DARK_TURQUOISE",
    4: "SOLID-RED",
    5: "SOLID-DARK_PINK",
    6: "SOLID-BROWN",
    7: "SOLID-LIGHT_GRAY",
    8: "SOLID-DARK_GRAY",
    9: "SOLID-LIGHT_BLUE",
    10: "SOLID-BRIGHT_GREEN",
    11: "SOLID-LIGHT_TURQUOISE",
    12: "SOLID-SALMON",
    13: "SOLID-PINK",
    14: "SOLID-YELLOW",
    15: "SOLID-WHITE",
    16: "SOLID-WHITE",
    17: "SOLID-LIGHT_GREEN",
    18: "SOLID-LIGHT_YELLOW",
    19: "SOLID-TAN",
    20: "SOLID-LIGHT_VIOLET",
    21: "MILKY-GLOW_IN_DARK_OPAQUE",
    22: "SOLID-PURPLE",
    23: "SOLID-DARK_BLUE_VIOLET",
    24: "SOLID-BLACK",
    25: "SOLID-ORANGE",
    26: "SOLID-MAGENTA",
    27: "SOLID-LIME",
    28: "SOLID-DARK_TAN",
    29: "SOLID-BRIGHT_PINK",
    30: "SOLID-MEDIUM_LAVENDER",
    31: "SOLID-LAVENDER",
    32: "SOLID-BLACK",
    33: "TRANS-DARK_BLUE",
    34: "TRANS-GREEN",
    35: "TRANS-BRIGHT_GREEN",
    36: "TRANS-RED",
    37: "TRANS-DARK_PINK",
    38: "TRANS-NEON_ORANGE",
    39: "TRANS-AQUA",
    40: "TRANS-BROWN",
    41: "TRANS-MEDIUM_BLUE",
    42: "TRANS-NEON_GREEN",
    43: "TRANS-LIGHT_BLUE",
    44: "TRANS-LIGHT_PURPLE",
    45: "TRANS-PINK",
    46: "TRANS-YELLOW",
    47: "TRANS-CLEAR",
    52: "TRANS-PURPLE",
    54: "TRANS-NEON_YELLOW",
    57: "TRANS-ORANGE",
    60: "CHROME-ANTIQUE_BRASS",
    61: "CHROME-BLUE",
    62: "CHROME-GREEN",
    63: "CHROME-PINK",
    64: "CHROME-BLACK",
    65: "RUBBER-YELLOW",
    66: "RUBBER-TRANS_YELLOW",
    67: "RUBBER-TRANS_CLEAR",
    68: "SOLID-VERY_LIGHT_ORANGE",
    69: "SOLID-LIGHT_PURPLE",
    70: "SOLID-REDDISH_BROWN",
    71: "SOLID-LIGHT_BLUISH_GRAY",
    72: "SOLID-DARK_BLUISH_GRAY",
    73: "SOLID-MEDIUM_BLUE",
    74: "SOLID-MEDIUM_GREEN",
    75: "SPECKLE-BLACK_COPPER",
    76: "SPECKLE-DBGRAY_SILVER",
    77: "SOLID-LIGHT_PINK",
    78: "SOLID-LIGHT_NOUGAT",
    79: "MILKY-WHITE",
    80: "METAL-SILVER",
    81: "METAL-GREEN",
    82: "METAL-GOLD",
    83: "METAL-BLACK",
    84: "SOLID-MEDIUM_NOUGAT",
    85: "SOLID-DARK_PURPLE",
    86: "SOLID-LIGHT_BROWN",
    87: "PEARL-DARK_GRAY",
    89: "SOLID-BLUE_VIOLET",
    92: "SOLID-NOUGAT",
    100: "SOLID-LIGHT_SALMON",
    110: "SOLID-VIOLET",
    112: "SOLID-MEDIUM_VIOLET",
    114: "GLITTER-TRANS_DARK_PINK",
    115: "SOLID-MEDIUM_LIME",
    117: "GLITTER-TRANS_CLEAR",
    118: "SOLID-AQUA",
    120: "SOLID-LIGHT_LIME",
    125: "SOLID-LIGHT_ORANGE",
    128: "SOLID-DARK_NOUGAT",
    129: "GLITTER-TRANS_PURPLE",
    132: "SPECKLE-BLACK_SILVER",
    133: "SPECKLE-BLACK_GOLD",
    134: "PEARL-COPPER",
    135: "PEARL-LIGHT_GRAY",
    137: "PEARL-SAND_BLUE",
    142: "PEARL-LIGHT_GOLD",
    147: "PEARL-FLAT_DARK_GOLD",
    148: "PEARL-DARK_GRAY",
    150: "PEARL-VERY_LIGHT_GRAY",
    151: "SOLID-VERY_LIGHT_BLUISH_GRAY",
    158: "TRANS-DARK_PINK",
    176: "PEARL-BIONICLE_COPPER",
    178: "PEARL-BIONICLE_GOLD",
    179: "PEARL-FLAT_SILVER",
    183: "PEARL-WHITE",
    184: "PEARL-RED",
    185: "PEARL-BLUE",
    186: "PEARL-GREEN",
    187: "PEARL-BROWN",
    189: "PEARL-REDDISH_GOLD",
    191: "SOLID-BRIGHT_LIGHT_ORANGE",
    200: "METAL-GREEN",
    212: "SOLID-BRIGHT_LIGHT_BLUE",
    216: "SOLID-RUST",
    218: "SOLID-CLIKITS_LAVENDER",
    219: "SOLID-LILAC",
    226: "SOLID-BRIGHT_LIGHT_YELLOW",
    227: "TRANS-LIGHT_BRIGHT_GREEN",
    231: "TRANS-LIGHT_ORANGE",
    232: "SOLID-SKY_BLUE",
    234: "TRANS-FIRE_YELLOW",
    256: "RUBBER-BLACK",
    272: "SOLID-DARK_BLUE",
    273: "RUBBER-BLUE",
    284: "TRANS-LIGHT_PURPLE",
    285: "TRANS-LIGHT_GREEN",
    288: "SOLID-DARK_GREEN",
    293: "TRANS-LIGHT_BLUE_VIOLET",
    294: "MILKY-GLOW_IN_DARK_TRANS",
    295: "SOLID-FLAMINGO_PINK",
    297: "PEARL-GOLD",
    300: "METAL-COPPER",
    302: "GLITTER-TRANS_LIGHT_BLUE",
    308: "SOLID-DARK_BROWN",
    313: "SOLID-MAERSK_BLUE",
    320: "SOLID-DARK_RED",
    321: "SOLID-DARK_AZURE",
    322: "SOLID-MEDIUM_AZURE",
    323: "SOLID-LIGHT_AQUA",
    324: "RUBBER-RED",
    326: "SOLID-YELLOWISH_GREEN",
    329: "MILKY-GLOW_IN_DARK_WHITE",
    330: "SOLID-OLIVE_GREEN",
    334: "CHROME-GOLD",
    335: "SOLID-SAND_RED",
    339: "GLITTER-TRANS_NEON_GREEN",
    341: "GLITTER-TRANS_ORANGE",
    350: "RUBBER-ORANGE",
    351: "SOLID-MEDIUM_DARK_PINK",
    353: "SOLID-CORAL",
    360: "SATIN-TRANS_CLEAR",
    362: "SATIN-TRANS_LIGHT_BLUE",
    363: "SATIN-TRANS_BROWN",
    364: "SATIN-TRANS_DARK_PINK",
    365: "SATIN-TRANS_PURPLE",
    366: "SOLID-EARTH_ORANGE",
    367: "SATIN-TRANS_BRIGHT_GREEN",
    368: "SOLID-NEON_YELLOW",
    370: "SOLID-MEDIUM_BROWN",
    371: "SOLID-MEDIUM_TAN",
    373: "SOLID-SAND_PURPLE",
    375: "RUBBER-LIGHT_GRAY",
    378: "SOLID-SAND_GREEN",
    379: "SOLID-SAND_BLUE",
    383: "CHROME-SILVER",
    402: "SOLID-REDDISH_ORANGE",
    406: "RUBBER-DARK_BLUE",
    422: "SOLID-SIENNA",
    423: "SOLID-UMBER",
    430: "SOLID-WARM_PINK",
    449: "RUBBER-PURPLE",
    450: "SOLID-FABULAND_BROWN",
    462: "SOLID-MEDIUM_ORANGE",
    484: "SOLID-DARK_ORANGE",
    490: "RUBBER-LIME",
    493: "METAL-MAGNET",
    494: "METAL-SILVER",
    495: "METAL-COPPER",
    496: "RUBBER-LIGHT_BLUISH_GRAY",
    503: "SOLID-VERY_LIGHT_GRAY",
    504: "RUBBER-FLAT_SILVER",
    507: "SOLID-LIGHT_ORANGE_BROWN",
    508: "SOLID-FABULAND_RED",
    509: "SOLID-FABULAND_ORANGE",
    510: "SOLID-FABULAND_LIME",
    511: "RUBBER-WHITE",
    10002: "RUBBER-GREEN",
    10010: "RUBBER-BRIGHT_GREEN",
    10019: "RUBBER-TAN",
    10022: "SOLID-YELLOWISH_DARK_PINK",
    10026: "RUBBER-MAGENTA",
    10030: "RUBBER-MEDIUM_LAVENDER",
    10031: "RUBBER-LAVENDER",
    10035: "RUBBER-TRANS_BRIGHT_GREEN",
    10036: "RUBBER-TRANS_RED",
    10043: "RUBBER-TRANS_LIGHT_BLUE",
    10045: "METAL-LIGHT_BLUE",
    10046: "METAL-PINK",
    10047: "TRANS-STICKER",
    10049: "METAL-LIGHT_PINK",
    10070: "RUBBER-REDDISH_BROWN",
    10073: "RUBBER-MEDIUM_BLUE",
    10078: "SOLID-LIGHT_NOUGAT",
    10135: "RUBBER-PEARL_LIGHT_GRAY",
    10179: "PEARL-BIONICLE_SILVER",
    10226: "RUBBER-BRIGHT_LIGHT_YELLOW",
    10297: "RUBBER-PEARL_GOLD",
    10308: "RUBBER-DARK_BROWN",
    10320: "RUBBER-DARK_RED",
    10321: "RUBBER-DARK_AZURE",
    10322: "RUBBER-MEDIUM_AZURE",
    10323: "RUBBER-LIGHT_AQUA",
    10351: "GLITTER-TRANS_BRIGHT_GREEN",
    10366: "SATIN-TRANS_DARK_BLUE",
    10375: "TRANS-BLACK",
    10378: "RUBBER-SAND_GREEN",
    10484: "RUBBER-DARK_ORANGE",
    11015: "TRANS-WHITE",
    11019: "TRANS-TAN",
    20000: "CANVAS-BLACK",
    20001: "CANVAS-BLUE",
    20004: "CANVAS-RED",
    20010: "CANVAS-BRIGHT_GREEN",
    20015: "CANVAS-WHITE",
    20086: "CANVAS-LIGHT_BROWN",
    20288: "CANVAS-DARK_GREEN",
    100003: "SOLID-DARK_YELLOW",
    100007: "SOLID-NEON_ORANGE",
    100009: "SOLID-NEON_GREEN",
    100040: "TRANS-BLACK",
    100167: "SOLID-REDDISH_ORANGE",
    100238: "PEARL-BIONICLE_GOLD",
    100241: "SOLID-MEDIUM_TAN",
    100249: "PEARL-REDDISH_COPPER",
    100255: "PEARL-BROWN",
    100273: "PEARL-BIONICLE_COPPER",
    1009021: "TRANS-LUMINOUS_SOFT_WHITE",
    1009024: "TRANS-LUMINOUS_SOFT_AMBER",
    1009025: "TRANS-LUMINOUS_SOFT_GREEN",
    1009027: "TRANS-LUMINOUS_WHITE",
}

_WHITE = (1.0, 1.0, 1.0, 1.0)
_TRANSMISSIVE_AO_ATTRS = {"pass_index": 2}
_LOCAL_AO_ATTRS = {"pass_index": 3}
_TRANSMISSIVE_LOCAL_AO_ATTRS = {"pass_index": 4}
MATERIAL_KIND_PROP = "_bricknet_render_material_kind"
CATALOG_MATERIAL_KIND = "catalog"
CATALOG_MATERIAL_NAME_PROP = "_bricknet_render_catalog_material"
BASE_COLOR_NODE_PROP = "_bricknet_render_base_color"


def _ao_attrs(*, transmissive=False, local=False):
    if local:
        return _TRANSMISSIVE_LOCAL_AO_ATTRS if transmissive else _LOCAL_AO_ATTRS
    if transmissive:
        return _TRANSMISSIVE_AO_ATTRS
    return None


def _build_material(name, shader_func, *, attrs, kwargs, viewport_color):
    existing = bpy.data.materials.get(name)
    if (
        existing is not None
        and existing.get(MATERIAL_KIND_PROP) == CATALOG_MATERIAL_KIND
        and existing.get(CATALOG_MATERIAL_NAME_PROP) == name
    ):
        return existing
    if existing is not None:
        raise RuntimeError("material name %s is already in use" % name)
    material = node_utils.shaderfunc_to_material(shader_func, name=name, **kwargs)
    material[MATERIAL_KIND_PROP] = CATALOG_MATERIAL_KIND
    material[CATALOG_MATERIAL_NAME_PROP] = name
    for key, value in attrs.items():
        setattr(material, key, value)
    material.diffuse_color = viewport_color
    return material


def shader_material(name, shader_func, *, viewport_color, attrs=None, **kwargs):
    return (
        name,
        partial(
            _build_material,
            name,
            shader_func,
            attrs=attrs or {},
            kwargs=kwargs,
            viewport_color=viewport_color,
        ),
    )


def _rgb_node(nw: NodeWrangler, name: str, color, *, base_color=False):
    node = nw.new_node(Nodes.RGB, label=name)
    node.outputs["Color"].default_value = color
    if base_color:
        node[BASE_COLOR_NODE_PROP] = True
    return node


def _principled_kwargs(*, subsurface, specular=0.5):
    return {
        "Subsurface Weight": subsurface,
        "Specular IOR Level": specular,
        "IOR": ng.SPECULAR_IOR,
    }


def _milky_sss_kwargs(nw, color_socket, radius_socket, label_prefix):
    fitted = ng.burley_sss_radius_fitting(nw, color_socket, radius_socket, label_prefix)
    return {"Subsurface Radius": fitted, "Subsurface Scale": ng.MILKY_SUBSURFACE_SCALE}


def _assign_rgb_curve(node, channels) -> None:
    for curve_index, points in enumerate(channels, start=1):
        node_utils.assign_curve(node.mapping.curves[curve_index], points)
    node.mapping.update()


def _material_group(nw: NodeWrangler, group_name: str, input_kwargs: dict):
    node_group = NODE_GROUPS[group_name]()
    return nw.new_node(node_group, label=node_group.name, input_kwargs=input_kwargs)


def shader_color_group(nw, *, group_name, color, output_socket="BSDF"):
    rgb = _rgb_node(nw, "RGB", color, base_color=True)
    inputs = {"BaseColor": rgb.outputs["Color"]}
    group = _material_group(nw, group_name, inputs)
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": group.outputs[output_socket]})


def shader_transparent(nw):
    transparent = nw.new_node("ShaderNodeBsdfTransparent", label="TransparentBSDF")
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": transparent.outputs["BSDF"]})


def shader_trans(nw, *, color, group_name="TRANS-GROUP_BASE", white=_WHITE):
    rgb = _rgb_node(nw, "RGB", color, base_color=True)
    rgb_white = rgb if white is None else _rgb_node(nw, "RGB_White", white)
    group = _material_group(
        nw,
        group_name,
        {"WhiteColor": rgb_white.outputs["Color"], "Color": rgb.outputs["Color"]},
    )
    nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": group.outputs["Shader"], "Volume": group.outputs["Volume"]}
    )


def shader_glitter(nw, *, color, chip, white, group_name="GLITTER-TRANS-GROUP"):
    rgb = _rgb_node(nw, "RGB", color, base_color=True)
    rgb_chip = _rgb_node(nw, "RGB_Chip", chip)
    rgb_white = rgb if white is None else _rgb_node(nw, "RGB_White", white)
    group = _material_group(
        nw,
        group_name,
        {
            "WhiteColor": rgb_white.outputs["Color"],
            "Color": rgb.outputs["Color"],
            "ChipColor": rgb_chip.outputs["Color"],
        },
    )
    nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": group.outputs["Shader"], "Volume": group.outputs["Volume"]}
    )


def shader_satin(nw, *, color, scale, group_name="SATIN-GROUP"):
    rgb_white = _rgb_node(nw, "RGB_White", (1.0, 0.9, 0.9, 1.0))
    rgb = _rgb_node(nw, "RGB", color, base_color=True)
    group = _material_group(
        nw,
        group_name,
        {
            "VoronoiTextureScale": scale,
            "WhiteColor": rgb_white.outputs["Color"],
            "Color": rgb.outputs["Color"],
        },
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": group.outputs["BSDF"]})


def shader_param_group(nw, *, group_name, params):
    group = _material_group(nw, group_name, params)
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": group.outputs["BSDF"]})


def shader_speckle(nw, *, base, chip, xoffset):
    rgb = _rgb_node(nw, "RGB", base, base_color=True)
    rgb_second = _rgb_node(nw, "RGB_Second", chip)
    group = _material_group(
        nw,
        "SPECKLE-GROUP",
        {
            "BaseColor": rgb.outputs["Color"],
            "ChipColor": rgb_second.outputs["Color"],
            "XOffset": xoffset,
        },
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": group.outputs["Shader"]})


def shader_milky_glow(nw, *, color, glow_color, subsurface, transmission, radius, min_roughness):
    rgb = _rgb_node(nw, "RGB", color, base_color=True)
    min_roughness_node = nw.new_node(Nodes.Value, label="MinRoughness")
    min_roughness_node.outputs["Value"].default_value = min_roughness
    normal = ng._rounded_normal(nw, "MilkyGlowNormal")
    sss_kwargs = {}
    if subsurface:
        sss_kwargs = _milky_sss_kwargs(nw, rgb.outputs["Color"], radius, "MilkyGlow")
    bsdf = ng.new_principled(
        nw,
        attrs=ng.GGX_BURLEY,
        input_kwargs={
            **_principled_kwargs(subsurface=subsurface),
            **sss_kwargs,
            "Base Color": rgb.outputs["Color"],
            "Roughness": min_roughness_node.outputs["Value"],
            "Normal": normal,
        },
        label="PrincipledBSDF",
    )
    shader = bsdf
    if transmission:
        glass = ng.fresnel_glass_mix(
            nw,
            rgb.outputs["Color"],
            min_roughness_node.outputs["Value"],
            normal,
            "MilkyGlow",
        )
        shader = nw.new_node(
            Nodes.MixShader,
            input_kwargs={"Factor": transmission, 1: bsdf, 2: glass},
            label="MilkyGlowGlassMix",
        )
    glow = _rgb_node(nw, "RGB_GlowDark", glow_color)
    emission = nw.new_node(
        Nodes.Emission, input_kwargs={"Color": glow.outputs["Color"], "Strength": 0.4}, label="Emission"
    )
    mix = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": 0.2, 1: shader, 2: emission.outputs["Emission"]},
        label="MixShader",
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": mix.outputs["Shader"]})


def shader_milky_white(nw):
    rgb = _rgb_node(nw, "RGB", (0.97, 0.96, 0.94, 1.0), base_color=True)
    normals1 = ng.rounded_normal_inputs(nw, "MilkyWhiteRoundingEdgeNormal")
    bsdf1 = ng.new_principled(
        nw,
        attrs=ng.GGX_BURLEY,
        input_kwargs={
            **_principled_kwargs(subsurface=1.0),
            **_milky_sss_kwargs(nw, rgb.outputs["Color"], (0.75, 0.75, 0.75), "MilkyWhite"),
            "Base Color": rgb.outputs["Color"],
            "Roughness": 0.2,
            **normals1,
        },
        label="PrincipledBSDF1",
    )
    glass = ng.fresnel_glass_mix(nw, rgb.outputs["Color"], 0.2, normals1["Normal"], "MilkyWhite")
    shader1 = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": 0.1, 1: bsdf1, 2: glass},
        label="MilkyWhiteGlassMix",
    )
    geometry = nw.new_node(Nodes.NewGeometry, label="Geometry")
    voronoi1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": geometry.outputs["Position"], "Scale": 150.0},
        label="VoronoiTexture1",
    )
    curves1 = nw.new_node(Nodes.RGBCurve, input_kwargs={"Color": voronoi1.outputs["Distance"]}, label="RGBCurves1")
    _assign_rgb_curve(curves1, (curve_data.MILKY_WHITE_MASK_CURVE,) * 3)
    coord_fix = nw.new_node(Nodes.TextureCoord, label="Milky White Texture Coordinates")
    voronoi2 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={"Vector": coord_fix.outputs["Object"], "Scale": 100.0},
        label="VoronoiTexture2",
    )
    curves2 = nw.new_node(Nodes.RGBCurve, input_kwargs={"Color": voronoi2.outputs["Color"]}, label="RGBCurves2")
    _assign_rgb_curve(
        curves2,
        (
            curve_data.MILKY_WHITE_FLAKE_R_CURVE,
            curve_data.MILKY_WHITE_FLAKE_G_CURVE,
            curve_data.MILKY_WHITE_FLAKE_B_CURVE,
        ),
    )
    bsdf2 = ng.new_principled(
        nw,
        attrs=ng.GGX_BURLEY,
        input_kwargs={
            **_principled_kwargs(subsurface=0.0, specular=1.0),
            "Roughness": 0.3,
            "Base Color": curves2.outputs["Color"],
            **normals1,
        },
        label="PrincipledBSDF2",
    )
    mix = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": curves1.outputs["Color"], 1: shader1, 2: bsdf2.outputs["BSDF"]},
        label="MixShader",
    )
    nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": mix.outputs["Shader"]})


def _color_group_material(name, color, group_name, *, attrs=None, output_socket="BSDF"):
    return shader_material(
        name,
        shader_color_group,
        viewport_color=color,
        attrs=attrs,
        group_name=group_name,
        color=color,
        output_socket=output_socket,
    )


def solid(name, color):
    return _color_group_material(name, color, "SOLID-GROUP", output_socket="Shader")


def rubber(name, color):
    return _color_group_material(name, color, "RUBBER-GROUP")


def rubber_trans(name, color):
    return _color_group_material(name, color, "RUBBER-TRANS-GROUP", attrs=_ao_attrs(transmissive=True))


def chrome(name, color, *, group_name="CHROME-GROUP"):
    return _color_group_material(name, color, group_name)


def metal(name, color):
    return _color_group_material(
        name,
        color,
        "METAL-GROUP",
        output_socket="Shader",
    )


def trans(name, color, **kwargs):
    return shader_material(
        name,
        shader_trans,
        viewport_color=color,
        attrs=_ao_attrs(transmissive=True),
        color=color,
        **kwargs,
    )


def trans_neon(name, color):
    return trans(name, color, group_name="TRANS-NEON-GROUP", white=None)


def glitter(
    name,
    color,
    *,
    chip=_WHITE,
    white=_WHITE,
    group_name="GLITTER-TRANS-GROUP",
):
    return shader_material(
        name,
        shader_glitter,
        viewport_color=color,
        attrs=_ao_attrs(transmissive=True),
        color=color,
        chip=chip,
        white=white,
        group_name=group_name,
    )


def satin(name, color, *, scale=300.0, group_name="SATIN-GROUP"):
    return shader_material(
        name,
        shader_satin,
        viewport_color=color,
        attrs=_ao_attrs(transmissive=True, local=True),
        color=color,
        scale=scale,
        group_name=group_name,
    )


def pearl_color(name, color, *, subsurface=1.0, clearcoat=0.0, metallic=0.0, group_name="PEARL-GROUP"):
    return param_group(
        name,
        group_name,
        {"Subsurface": subsurface, "Clearcoat": clearcoat, "Metallic": metallic, "Color": color},
        viewport_color=color,
        attrs=_ao_attrs(local=True),
    )


def param_group(name, group_name, params, *, viewport_color, attrs=None):
    return shader_material(
        name,
        shader_param_group,
        viewport_color=viewport_color,
        attrs=attrs,
        group_name=group_name,
        params=dict(params),
    )


def pearl_shine(
    name,
    base,
    shine,
    *,
    radius=(0.3, 0.3, 0.3),
    roughness=0.29,
    metallic=0.1,
    layer=0.95,
):
    return param_group(
        name,
        "SHINE_GROUP",
        {
            "BaseColor": base,
            "ShineColor": shine,
            "SubsurfaceRadius": radius,
            "SubsurfaceRoughness": roughness,
            "Metallic": metallic,
            "LayerWeight": layer,
        },
        viewport_color=base,
        attrs=_ao_attrs(local=True),
    )


def pearl_principled(
    name,
    base,
    *,
    radius=(0.3, 0.3, 0.3),
    roughness=0.29,
    clearcoat=0.3,
    metallic=0.1,
):
    return param_group(
        name,
        "PRINCIPLED-GROUP",
        {
            "BaseColor": base,
            "SubsurfaceRadius": radius,
            "SubsurfaceRoughness": roughness,
            "Clearcoat": clearcoat,
            "Metallic": metallic,
        },
        viewport_color=base,
        attrs=_ao_attrs(local=True),
    )


def speckle(name, base, *, chip, xoffset):
    return shader_material(
        name,
        shader_speckle,
        viewport_color=base,
        base=base,
        chip=chip,
        xoffset=xoffset,
    )


def milky_glow(
    name,
    color,
    *,
    glow_color,
    subsurface,
    transmission,
    min_roughness,
    radius=(1.0, 1.0, 1.0),
    local_ao=False,
):
    return shader_material(
        name,
        shader_milky_glow,
        viewport_color=color,
        attrs=_ao_attrs(transmissive=bool(transmission), local=local_ao),
        color=color,
        glow_color=glow_color,
        subsurface=subsurface,
        transmission=transmission,
        radius=radius,
        min_roughness=min_roughness,
    )


def _scene_linear(value: int):
    """Convert a 24-bit sRGB value to Blender scene-linear RGBA."""

    c = Color(tuple(((value >> shift) & 0xFF) / 255.0 for shift in (16, 8, 0)))
    return (*c.from_srgb_to_scene_linear(), 1.0)


SIMPLE_MATERIAL_SPECS = (
    satin("SATIN-TRANS_CLEAR", (1.0, 1.0, 0.95, 1.0), scale=100.0),
    satin(
        "SATIN-TRANS_BROWN",
        (0.38824, 0.37255, 0.32157, 1.0),
        scale=350.0,
        group_name="SATIN-TRANS-GROUP",
    ),
    satin(
        "SATIN-TRANS_LIGHT_BLUE",
        (0.6014189, 0.9457989, 1.0, 1.0),
        scale=100.0,
        group_name="SATIN-TRANS-GROUP",
    ),
    satin(
        "SATIN-TRANS_DARK_PINK",
        (0.8117647, 0.154902, 0.6117647, 1.0),
    ),
    satin("SATIN-TRANS_BRIGHT_GREEN", (0.3, 0.8, 0.1, 1.0), scale=350.0),
    satin("SATIN-TRANS_DARK_BLUE", (0.0, 0.4, 0.9, 1.0)),
    satin("SATIN-TRANS_PURPLE", (0.23, 0.2, 1.0, 1.0)),
    trans("TRANS-ORANGE", (1.0, 0.6434769, 0.2, 1.0)),
    shader_material(
        "TRANS-STICKER",
        shader_transparent,
        viewport_color=(1.0, 1.0, 1.0, 0.0),
        attrs=_ao_attrs(transmissive=True),
    ),
    solid("SOLID-WHITE", (1.0, 0.941, 0.9, 1.0)),
    solid("SOLID-DARK_BLUISH_GRAY", (0.151228, 0.1601239, 0.1690196, 1.0)),
    solid("SOLID-LIGHT_GRAY", (0.441, 0.45, 0.423, 1.0)),
    solid("SOLID-VERY_LIGHT_BLUISH_GRAY", (0.8938432, 0.9020173, 0.92, 1.0)),
    solid("SOLID-LIGHT_BLUISH_GRAY", (0.4838704, 0.4986897, 0.509804, 1.0)),
    solid("SOLID-DARK_GRAY", (0.2, 0.1952377, 0.1714286, 1.0)),
    solid("SOLID-VERY_LIGHT_GRAY", (0.62, 0.5985065, 0.5703999, 1.0)),
    solid("SOLID-BLACK", (0.005, 0.005, 0.005, 1.0)),
    solid("SOLID-BLUE", (0.0, 0.1053925, 0.5, 1.0)),
    solid("SOLID-GREEN", (0.0, 0.26, 0.08233331, 1.0)),
    solid("SOLID-YELLOW", (1.0, 0.6666667, 0.0, 1.0)),
    solid("SOLID-MAGENTA", (0.4117647, 0.01176471, 0.1843137, 1.0)),
    solid("SOLID-RED", (0.6, 0.007843138, 0.007843138, 1.0)),
    chrome("CHROME-SILVER", (0.875, 0.875, 0.875, 1.0)),
    chrome("CHROME-ANTIQUE_BRASS", (1.0, 0.8438869, 0.2999606, 1.0), group_name="CHROME-ANTIQUE-GROUP"),
    chrome("CHROME-BLUE", (0.3032564, 0.448617, 0.7038274, 1.0)),
    chrome("CHROME-GOLD", (1.0, 0.8438869, 0.2999606, 1.0)),
    chrome("CHROME-GREEN", (0.03282344, 0.5000563, 0.1112415, 1.0)),
    chrome("CHROME-PINK", (0.7225216, 0.04129564, 0.4954462, 1.0)),
    chrome("CHROME-BLACK", (0.02573991, 0.03861832, 0.0469642, 1.0)),
    metal("METAL-MAGNET", (0.2, 0.1952377, 0.1714286, 1.0)),
    metal("METAL-GREEN", (0.5382832, 0.52026, 0.1778194, 1.0)),
    metal("METAL-COPPER", (0.5686275, 0.2, 0.04313726, 1.0)),
    metal("METAL-SILVER", (0.748099, 0.7485667, 0.75, 1.0)),
    metal("METAL-GOLD", (0.8, 0.5803623, 0.2009881, 1.0)),
    metal("METAL-LIGHT_BLUE", _scene_linear(0x97CBD9)),
    metal("METAL-PINK", _scene_linear(0xAD659A)),
    metal("METAL-LIGHT_PINK", _scene_linear(0xFECCCF)),
    solid("SOLID-DARK_RED", (0.2392157, 0.01176471, 0.01176471, 1.0)),
    solid("SOLID-REDDISH_ORANGE", (1.0, 0.11583333, 0.0, 1.0)),
    solid("SOLID-SALMON", (1.0, 0.26, 0.21, 1.0)),
    solid("SOLID-LIGHT_SALMON", (1.0, 0.6156863, 0.5294118, 1.0)),
    solid("SOLID-CORAL", (1.0, 0.2117, 0.2117, 1.0)),
    solid("SOLID-SAND_RED", (0.4607843, 0.2256863, 0.2156863, 1.0)),
    solid("SOLID-REDDISH_BROWN", (0.16, 0.04662154, 0.02116923, 1.0)),
    solid("SOLID-RUST", (0.4203726, 0.03529412, 0.03137255, 1.0)),
    solid("SOLID-FABULAND_RED", (0.654902, 0.1294118, 0.04705882, 1.0)),
    solid("SOLID-BROWN", (0.16, 0.05356308, 0.02116923, 1.0)),
    solid("SOLID-SIENNA", (0.51, 0.13002, 0.04099999, 1.0)),
    solid("SOLID-UMBER", (0.13, 0.05862154, 0.02616923, 1.0)),
    solid("SOLID-DARK_TAN", (0.3, 0.2168, 0.108, 1.0)),
    solid("SOLID-TAN", (0.82, 0.6915452, 0.3918174, 1.0)),
    solid("SOLID-MEDIUM_TAN", (0.84, 0.53, 0.265, 1.0)),
    solid("SOLID-LIGHT_NOUGAT", (1.0, 0.6626376, 0.4513225, 1.0)),
    solid("SOLID-NOUGAT", (0.86, 0.3182, 0.1376, 1.0)),
    solid("SOLID-LIGHT_FLESH", (1.0, 0.6626376, 0.4513225, 1.0)),
    solid("SOLID-FLESH", (0.86, 0.3182, 0.1376, 1.0)),
    solid("SOLID-DARK_BROWN", (0.05532353, 0.02406236, 0.01066471, 1.0)),
    solid("SOLID-MEDIUM_NOUGAT", (0.6, 0.2302, 0.08399999, 1.0)),
    solid("SOLID-MEDIUM_DARK_FLESH", (0.6, 0.2302, 0.08399999, 1.0)),
    solid("SOLID-DARK_FLESH", (0.28, 0.09940001, 0.02800001, 1.0)),
    solid("SOLID-FABULAND_ORANGE", (0.5019608, 0.2068026, 0.03592158, 1.0)),
    solid("SOLID-EARTH_ORANGE", (0.6, 0.1947541, 0.03454056, 1.0)),
    solid("SOLID-LIGHT_ORANGE_BROWN", (0.6, 0.1947541, 0.03454056, 1.0)),
    solid("SOLID-DARK_NOUGAT", (0.5, 0.1607541, 0.02814056, 1.0)),
    solid("SOLID-DARK_ORANGE", (0.4, 0.106, 0.04000001, 1.0)),
    solid("SOLID-ORANGE", (1.0, 0.2083333, 0.0, 1.0)),
    solid("SOLID-FABULAND_BROWN", (0.5019608, 0.1677386, 0.07529411, 1.0)),
    solid("SOLID-MEDIUM_ORANGE", (0.9619608, 0.3936641, 0.03847845, 1.0)),
    solid("SOLID-LIGHT_ORANGE", (1.0, 0.4553333, 0.14, 1.0)),
    solid("SOLID-BRIGHT_LIGHT_YELLOW", (1.0, 0.84, 0.2, 1.0)),
    solid("SOLID-LIGHT_LIME", (0.678, 0.619, 0.023, 1.0)),
    solid("SOLID-YELLOWISH_GREEN", (0.7819876, 1.0, 0.35, 1.0)),
    solid("SOLID-BRIGHT_LIGHT_ORANGE", (1.0, 0.4853743, 0.0, 1.0)),
    solid("SOLID-MEDIUM_LIME", (0.56, 0.68, 0.065, 1.0)),
    solid("SOLID-OLIVE_GREEN", (0.2392157, 0.2364242, 0.07172547, 1.0)),
    solid("SOLID-DARK_GREEN", (0.0, 0.05882353, 0.02745098, 1.0)),
    solid("SOLID-LIGHT_YELLOW", (1.0, 0.7938562, 0.3490196, 1.0)),
    solid("SOLID-BRIGHT_GREEN", (0.02987802, 0.35, 0.07256099, 1.0)),
    solid("SOLID-LIME", (0.8039, 1.0, 0.0039, 1.0)),
    solid("SOLID-FABULAND_LIME", (0.5490196, 0.7490196, 0.09411765, 1.0)),
    solid("SOLID-MEDIUM_GREEN", (0.3121311366558075, 0.8275709748268127, 0.39765840768814087, 1.0)),
    solid("SOLID-SAND_GREEN", (0.1861216, 0.3001961, 0.2013315, 1.0)),
    solid("SOLID-LIGHT_TURQUOISE", (0.0, 0.4, 0.3310347, 1.0)),
    solid("SOLID-DARK_TURQUOISE", (0.0, 0.3, 0.256522, 1.0)),
    solid("SOLID-AQUA", (0.5049412, 0.8705882, 0.7974588, 1.0)),
    solid("SOLID-LIGHT_AQUA", (0.7194118, 1.0, 0.9345294, 1.0)),
    solid("SOLID-LIGHT_GREEN", (0.4917253, 0.8, 0.4300703, 1.0)),
    solid("SOLID-DARK_BLUE", (0.01511765, 0.0280756, 0.07558824, 1.0)),
    solid("SOLID-MEDIUM_AZURE", (0.1239634, 0.5829892, 0.7219608, 1.0)),
    solid("SOLID-MEDIUM_BLUE", (0.1771999, 0.3786118, 0.6605883, 1.0)),
    solid("SOLID-MAERSK_BLUE", (0.1635955, 0.4398594, 0.5511765, 1.0)),
    solid("SOLID-BRIGHT_LIGHT_BLUE", (0.4014117, 0.6210371, 0.9098039, 1.0)),
    solid("SOLID-DARK_AZURE", (0.01815192, 0.2531261, 0.4911765, 1.0)),
    solid("SOLID-LIGHT_BLUE", (0.5776021, 0.7636749, 0.8605882, 1.0)),
    solid("SOLID-SAND_BLUE", (0.1834753, 0.2388641, 0.3011765, 1.0)),
    solid("SOLID-BLUE_VIOLET", (0.04647057, 0.107353, 0.4117647, 1.0)),
    solid("SOLID-DARK_BLUE_VIOLET", (0.0, 0.04500003, 0.3, 1.0)),
    solid("SOLID-VIOLET", (0.03921569, 0.03921569, 0.2901961, 1.0)),
    solid("SOLID-SKY_BLUE", (0.1378432, 0.5669582, 0.7019608, 1.0)),
    solid("SOLID-MEDIUM_VIOLET", (0.2036299, 0.270528, 0.7509804, 1.0)),
    solid("SOLID-LILAC", (0.1686275, 0.1607843, 0.7490196, 1.0)),
    solid("SOLID-DARK_PURPLE", (0.09216753, 0.04862745, 0.25, 1.0)),
    solid("SOLID-PURPLE", (0.1617649, 0.0245098, 0.25, 1.0)),
    solid("SOLID-LIGHT_PURPLE", (0.5088235, 0.06105882, 0.2935294, 1.0)),
    solid("SOLID-MEDIUM_LAVENDER", (0.3576361, 0.224, 0.56, 1.0)),
    solid("SOLID-LIGHT_VIOLET", (0.6392157, 0.6862745, 0.8509804, 1.0)),
    solid("SOLID-LAVENDER", (0.5546062, 0.4816567, 0.7219608, 1.0)),
    solid("SOLID-CLIKITS_LAVENDER", (0.6907843, 0.409804, 0.6986275, 1.0)),
    solid("SOLID-DARK_PINK", (0.6509804, 0.1882353, 0.3882353, 1.0)),
    solid("SOLID-MEDIUM_DARK_PINK", (0.6, 0.2509804, 0.3058824, 1.0)),
    solid("SOLID-FLAMINGO_PINK", (1.0, 0.4470588, 0.627451, 1.0)),
    solid("SOLID-BRIGHT_PINK", (1.0, 0.5803922, 0.8235294, 1.0)),
    solid("SOLID-PINK", (1.0, 0.6784314, 0.7882353, 1.0)),
    solid("SOLID-WARM_PINK", _scene_linear(0xF79B9B)),
    solid("SOLID-YELLOWISH_DARK_PINK", _scene_linear(0xD05098)),
    trans("TRANS-RED", (0.88, 0.0, 0.0, 1.0)),
    solid("SOLID-SAND_PURPLE", (0.2784314, 0.2196078, 0.2784314, 1.0)),
    solid("SOLID-NEON_ORANGE", (1.0, 0.205, 0.1, 1.0)),
    solid("SOLID-NEON_GREEN", (0.63, 0.9, 0.157, 1.0)),
    solid("SOLID-NEON_YELLOW", (0.85, 1.0, 0.1, 1.0)),
    solid("SOLID-DARK_YELLOW", (0.6709804, 0.3354902, 0.0, 1.0)),
    solid("SOLID-LIGHT_PINK", (0.9919999837875366, 0.7960000038146973, 0.8080000281333923, 1.0)),
    solid("SOLID-LIGHT_BROWN", (0.28, 0.09940001, 0.02800001, 1.0)),
    solid("SOLID-MEDIUM_BROWN", (0.22, 0.08040001, 0.03300001, 1.0)),
    solid("SOLID-VERY_LIGHT_ORANGE", (1.0, 0.6816186, 0.32, 1.0)),
    pearl_color(
        "PEARL-LIGHT_GRAY",
        (0.3752978, 0.3894884, 0.4, 1.0),
    ),
    pearl_color(
        "PEARL-VERY_LIGHT_GRAY",
        (0.5829391, 0.5487896, 0.5226755, 1.0),
    ),
    trans("TRANS-CLEAR", (1.0, 1.0, 1.0, 1.0)),
    trans_neon("TRANS-NEON_ORANGE", (0.97, 0.51895, 0.09700003, 1.0)),
    trans("TRANS-PURPLE", (0.3740637, 0.3116105, 0.82, 1.0)),
    trans("TRANS-BROWN", (0.4769998788833618, 0.4185596704483032, 0.374546080827713, 1.0)),
    trans("TRANS-BLACK", (0.4569998788833618, 0.4085596704483032, 0.44546080827713, 1.0)),
    trans("TRANS-LIGHT_ORANGE", (1.0, 0.8105064, 0.5, 1.0)),
    trans("TRANS-YELLOW", (1.0, 0.9137255, 0.0, 1.0)),
    trans("TRANS-FIRE_YELLOW", (1.0, 0.9843137, 0.5686275, 1.0)),
    trans_neon("TRANS-NEON_GREEN", (0.9128102, 0.96, 0.09600002, 1.0)),
    trans("TRANS-BRIGHT_GREEN", (0.2605008, 0.928, 0.1872534, 1.0)),
    trans("TRANS-GREEN", (0.0, 0.6, 0.1428191, 1.0)),
    trans_neon("TRANS-NEON_YELLOW", (0.96, 0.8912, 0.1344, 1.0)),
    trans("TRANS-DARK_BLUE", (0.16, 0.7200001, 1.0, 1.0)),
    trans("TRANS-LIGHT_BLUE", (0.6014189, 0.9457989, 1.0, 1.0)),
    trans("TRANS-AQUA", (0.8336286, 0.968052, 1.0, 1.0)),
    trans("TRANS-LIGHT_PURPLE", (0.72, 0.54, 0.6900001, 1.0)),
    trans("TRANS-DARK_PINK", (1.0, 0.5546498, 0.7865443, 1.0)),
    trans("TRANS-MEDIUM_BLUE", (0.4027255, 0.734454, 0.874, 1.0)),
    trans("TRANS-LIGHT_BLUE_VIOLET", (0.6588235, 0.772549, 0.8666667, 1.0)),
    trans("TRANS-PINK", (1.0, 0.677561, 0.7612647, 1.0)),
    trans("TRANS-LIGHT_GREEN", (0.58, 0.898, 0.67, 1.0)),
    trans("TRANS-LIGHT_BRIGHT_GREEN", (0.7077, 0.8489, 0.4234, 1.0)),
    trans("TRANS-WHITE", _scene_linear(0xF4F4F4)),
    trans("TRANS-TAN", _scene_linear(0xD7BA8C)),
    trans("TRANS-LUMINOUS_SOFT_WHITE", (0.984, 0.984, 0.984, 1.0)),
    trans("TRANS-LUMINOUS_SOFT_AMBER", (0.709, 0.329, 0.054, 1.0)),
    trans("TRANS-LUMINOUS_SOFT_GREEN", (0.023, 0.458, 0.176, 1.0)),
    trans("TRANS-LUMINOUS_WHITE", (0.984, 0.984, 0.984, 1.0)),
    pearl_color(
        "PEARL-LIGHT_GOLD",
        (0.9090196, 0.6370555, 0.3557199, 1.0),
        group_name="PEARL-FLAT-GROUP",
    ),
    pearl_color(
        "PEARL-BIONICLE_GOLD",
        (0.3960784, 0.167451, 0.0, 1.0),
    ),
    pearl_color(
        "PEARL-BIONICLE_SILVER",
        (0.2403527, 0.2169605, 0.221, 1.0),
        subsurface=0.515,
        clearcoat=1.0,
    ),
    pearl_color(
        "PEARL-BIONICLE_COPPER",
        (0.5686275, 0.1564706, 0.0473726, 1.0),
        subsurface=0.712,
        metallic=0.2,
    ),
    glitter(
        "GLITTER-TRANS_CLEAR",
        (0.996, 0.996, 0.996, 1.0),
        chip=(0.9959999918937683, 0.9959999918937683, 0.9959999918937683, 1.0),
    ),
    glitter(
        "GLITTER-TRANS_LIGHT_BLUE",
        (0.6014189, 0.9457989, 1.0, 1.0),
    ),
    glitter("GLITTER-TRANS_ORANGE", (1.0, 0.6434769, 0.2, 1.0)),
    glitter(
        "GLITTER-TRANS_BRIGHT_GREEN",
        (0.6862745, 0.9254902, 0.1803922, 1.0),
    ),
    glitter(
        "GLITTER-TRANS_PURPLE",
        (0.3740637, 0.3116105, 0.82, 1.0),
    ),
    glitter(
        "GLITTER-TRANS_DARK_PINK",
        (1.0, 0.5546498, 0.7865443, 1.0),
    ),
    speckle(
        "SPECKLE-BLACK_SILVER",
        (0.004999999888241291, 0.004999999888241291, 0.004999999888241291, 1.0),
        chip=(0.5382355451583862, 0.4785160720348358, 0.4785160720348358, 1.0),
        xoffset=0.50000214576721,
    ),
    glitter(
        "GLITTER-TRANS_NEON_GREEN",
        (0.9033018, 0.95, 0.09500002, 1.0),
        white=None,
        group_name="GLITTER-TRANS_NEON-GROUP",
    ),
    speckle(
        "SPECKLE-BLACK_GOLD",
        (0.004999999888241291, 0.004999999888241291, 0.004999999888241291, 1.0),
        chip=(0.55, 0.3728288, 0.03817804, 1.0),
        xoffset=0.44999998807907104,
    ),
    speckle(
        "SPECKLE-DBGRAY_SILVER",
        (0.14499999582767487, 0.1449524462223053, 0.1449524462223053, 1.0),
        chip=(0.8500000238418579, 0.8500000238418579, 0.8500000238418579, 1.0),
        xoffset=0.5400000214576721,
    ),
    rubber("RUBBER-BLACK", (0.01, 0.009653335, 0.0092, 1.0)),
    rubber("RUBBER-BLUE", (0.0, 0.1053925, 0.5, 1.0)),
    rubber("RUBBER-DARK_BLUE", (0.01511765, 0.0280756, 0.07558824, 1.0)),
    speckle(
        "SPECKLE-BLACK_COPPER",
        (0.004999999888241291, 0.004999999888241291, 0.004999999888241291, 1.0),
        chip=(0.2329999953508377, 0.10199999809265137, 0.023000000044703484, 1.0),
        xoffset=0.5400000214576721,
    ),
    rubber("RUBBER-FLAT_SILVER", (0.2393527, 0.2399605, 0.243, 1.0)),
    rubber("RUBBER-LIGHT_GRAY", (0.441, 0.45, 0.423, 1.0)),
    rubber("RUBBER-LIME", (0.406009, 0.549804, 0.07189742, 1.0)),
    rubber("RUBBER-LIGHT_BLUISH_GRAY", (0.4838704, 0.4986897, 0.509804, 1.0)),
    rubber("RUBBER-PURPLE", (0.1617649, 0.0245098, 0.25, 1.0)),
    rubber("RUBBER-GREEN", (0.0, 0.3568628, 0.1058824, 1.0)),
    rubber("RUBBER-SAND_GREEN", (0.1861216, 0.3001961, 0.2013315, 1.0)),
    rubber("RUBBER-MAGENTA", (0.4117647, 0.01176471, 0.1843137, 1.0)),
    rubber("RUBBER-LIGHT_NOUGAT", (1.0, 0.7882, 0.5843, 1.0)),
    rubber("RUBBER-MEDIUM_LAVENDER", (0.3576361, 0.224, 0.56, 1.0)),
    rubber("RUBBER-LAVENDER", (0.5546062, 0.4816567, 0.7219608, 1.0)),
    rubber("RUBBER-REDDISH_BROWN", (0.16, 0.04662154, 0.02116923, 1.0)),
    rubber("RUBBER-BRIGHT_LIGHT_YELLOW", (1.0, 0.84, 0.2, 1.0)),
    rubber("RUBBER-DARK_BROWN", (0.05532353, 0.02406236, 0.01066471, 1.0)),
    rubber("RUBBER-DARK_RED", (0.2392157, 0.01176471, 0.01176471, 1.0)),
    rubber("RUBBER-DARK_AZURE", (0.01815192, 0.2531261, 0.4911765, 1.0)),
    rubber("RUBBER-MEDIUM_AZURE", (0.1239634, 0.5829892, 0.7219608, 1.0)),
    rubber("RUBBER-DARK_ORANGE", (0.4, 0.106, 0.04000001, 1.0)),
    rubber("RUBBER-LIGHT_AQUA", (0.1239634, 0.5829892, 0.7219608, 1.0)),
    rubber_trans("RUBBER-TRANS_CLEAR", (0.9959999918937683, 0.9959999918937683, 0.9959999918937683, 1.0)),
    rubber_trans("RUBBER-TRANS_YELLOW", (1.0, 0.8767249, 0.07543677, 1.0)),
    rubber_trans("RUBBER-TRANS_BRIGHT_GREEN", (0.2784314, 1.0, 0.1960784, 1.0)),
    rubber_trans("RUBBER-TRANS_LIGHT_BLUE", (0.1960784, 0.6235294, 1.0, 1.0)),
    rubber_trans("RUBBER-TRANS_RED", (0.88, 0.0, 0.0, 1.0)),
    rubber("RUBBER-WHITE", (1.0, 0.941, 0.9, 1.0)),
    rubber("RUBBER-BRIGHT_GREEN", (0.02987802, 0.35, 0.07256099, 1.0)),
    rubber("RUBBER-MEDIUM_BLUE", (0.1771999, 0.3786118, 0.6605883, 1.0)),
    rubber("RUBBER-YELLOW", (1.0, 0.6666667, 0.0, 1.0)),
    rubber("RUBBER-RED", (0.6, 0.007843138, 0.007843138, 1.0)),
    rubber("RUBBER-ORANGE", (1.0, 0.2083333, 0.0, 1.0)),
    rubber("RUBBER-TAN", _scene_linear(0xD7BA8C)),
    rubber("RUBBER-PEARL_LIGHT_GRAY", _scene_linear(0xA0A0A0)),
    rubber("RUBBER-PEARL_GOLD", _scene_linear(0xAA7F2E)),
    # LDConfig's canvas colors have no corresponding catalog shader; use the solid surface fallback.
    solid("CANVAS-BLACK", _scene_linear(0x1B2A34)),
    solid("CANVAS-BLUE", _scene_linear(0x1E5AA8)),
    solid("CANVAS-RED", _scene_linear(0xB40000)),
    solid("CANVAS-BRIGHT_GREEN", _scene_linear(0x58AB41)),
    solid("CANVAS-WHITE", _scene_linear(0xF4F4F4)),
    solid("CANVAS-LIGHT_BROWN", _scene_linear(0x7B5D41)),
    solid("CANVAS-DARK_GREEN", _scene_linear(0x00451A)),
    pearl_principled(
        "PEARL-FLAT_FINE_SILVER",
        (0.45, 0.45, 0.43, 1.0),
        radius=(0.25, 0.25, 0.25),
        roughness=0.2,
        clearcoat=0.0,
        metallic=0.0,
    ),
    pearl_shine(
        "PEARL-FLAT_SILVER",
        (0.18, 0.193032, 0.2, 1.0),
        (0.6, 0.6, 0.6, 1.0),
        radius=(0.2, 0.2, 0.2),
        roughness=0.2,
        metallic=0.6,
        layer=0.92,
    ),
    pearl_shine(
        "PEARL-FLAT_LIGHT_GRAY",
        (0.607, 0.588, 0.47, 1.0),
        (0.858, 0.854, 0.819, 1.0),
        metallic=0.15,
    ),
    pearl_shine(
        "PEARL-DARK_GRAY",
        (0.03, 0.03, 0.03, 1.0),
        (0.29, 0.29, 0.29, 1.0),
    ),
    pearl_shine(
        "PEARL-FLAT_DARK_GOLD",
        (0.031063, 0.020854, 0.003782, 1.0),
        (0.854993, 0.630757, 0.258183, 1.0),
        roughness=0.2,
        metallic=0.25,
        layer=0.92,
    ),
    pearl_color(
        "PEARL-METALLIC_BRIGHT_RED",
        (0.9490196, 0.03921569, 0.0, 1.0),
        subsurface=0.4,
        group_name="PEARL-FLAT-GROUP",
    ),
    pearl_color(
        "PEARL-METALLIC_DARK_GREEN",
        (0.0, 0.2980392, 0.1372549, 1.0),
        subsurface=0.4,
        group_name="PEARL-FLAT-GROUP",
    ),
    pearl_shine(
        "PEARL-SAND_BLUE",
        (0.086, 0.18, 0.372, 1.0),
        (0.337164, 0.354644, 0.637597, 1.0),
        metallic=0.17,
        layer=0.92,
    ),
    pearl_shine(
        "PEARL-REDDISH_GOLD",
        (0.65, 0.329, 0.047, 1.0),
        (0.686686, 0.434154, 0.184475, 1.0),
        metallic=0.25,
    ),
    pearl_principled(
        "PEARL-BROWN",
        (0.07, 0.023, 0.0, 1.0),
    ),
    pearl_shine(
        "PEARL-GOLD",
        (0.197391, 0.095533, 0.009421, 1.0),
        (0.890005, 0.404788, 0.046014, 1.0),
        roughness=0.2,
        layer=0.75,
    ),
    pearl_shine(
        "PEARL-REDDISH_COPPER",
        (0.47034, 0.077218, 0.005009, 1.0),
        (0.47034, 0.097218, 0.005009, 1.0),
        radius=(0.15, 0.15, 0.15),
        roughness=0.2,
        metallic=0.35,
        layer=0.92,
    ),
    pearl_principled(
        "PEARL-COPPER",
        (0.282, 0.066, 0.017, 1.0),
        radius=(0.15, 0.15, 0.15),
        roughness=0.2,
        clearcoat=0.0,
        metallic=0.2,
    ),
    pearl_principled(
        "PEARL-RED",
        (0.558, 0.0, 0.0, 1.0),
    ),
    pearl_principled(
        "PEARL-WHITE",
        (0.878, 0.862, 0.796, 1.0),
        radius=(0.2, 0.2, 0.2),
    ),
    pearl_principled(
        "PEARL-GREEN",
        (0.0, 0.556, 0.235, 1.0),
    ),
    pearl_principled(
        "PEARL-BLUE",
        (0.0, 0.1, 0.558, 1.0),
        radius=(0.39, 0.4, 0.4),
        roughness=0.32,
    ),
    pearl_principled(
        "METAL-BLACK",
        (0.0, 0.0, 0.001, 1.0),
        radius=(0.4, 0.4, 0.4),
        roughness=0.32,
    ),
    shader_material(
        "MILKY-WHITE",
        shader_milky_white,
        viewport_color=(0.97, 0.96, 0.94, 1.0),
        attrs=_ao_attrs(transmissive=True, local=True),
    ),
    milky_glow(
        "MILKY-GLOW_IN_DARK_OPAQUE",
        (0.9300055, 0.8811104, 0.7774289, 1.0),
        glow_color=(0.2924865186214447, 0.4077401876449585, 0.2130640745162964, 1.0),
        subsurface=1.0,
        transmission=0.5,
        radius=(0.7, 0.7, 0.7),
        min_roughness=0.05000000149011612,
        local_ao=True,
    ),
    milky_glow(
        "MILKY-GLOW_IN_DARK_TRANS",
        (1.0, 0.9665679, 0.870479, 1.0),
        glow_color=(0.6935895, 0.9547417, 0.5656157, 1.0),
        subsurface=0.0,
        transmission=0.95,
        min_roughness=0.2,
    ),
    milky_glow(
        "MILKY-GLOW_IN_DARK_WHITE",
        (1.0, 1.0, 0.9, 1.0),
        glow_color=(0.2924865186214447, 0.4577401876449585, 0.2130640745162964, 1.0),
        subsurface=1.0,
        transmission=0.0,
        min_roughness=0.05,
        local_ao=True,
    ),
)


MATERIAL_SPECS = dict(SIMPLE_MATERIAL_SPECS)


def material_specs():
    """Return a copy of the material-builder registry."""

    return dict(MATERIAL_SPECS)


def build_material(name: str) -> bpy.types.Material | None:
    """Build a catalog material, or return ``None`` for an unknown name."""

    builder = MATERIAL_SPECS.get(name)
    return None if builder is None else builder()
