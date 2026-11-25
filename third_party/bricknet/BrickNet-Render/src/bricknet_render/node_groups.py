"""Define the reusable shader node groups used by the materials."""

import math

from . import curve_data, node_utils
from .node_utils import Nodes, NodeWrangler

_GGX = {"distribution": "GGX"}
GGX_BURLEY = {"distribution": "GGX", "subsurface_method": "BURLEY"}
ROUNDING_EDGE_NORMAL_RAYCAST_SAMPLES = 4
SURFACE_NORMAL_GROUP_NAME = "Mat_Surface Normal"
EDGE_ROUNDING = 0.05


def _rounded_normal(nw: NodeWrangler, label):
    return rounded_normal_inputs(nw, label)["Normal"]


def _color_multiply(nw: NodeWrangler, a, b, label):
    return nw.new_node(
        Nodes.Mix,
        input_kwargs={"Factor": 1.0, "A": a, "B": b},
        label=label,
        attrs={"data_type": "RGBA", "blend_type": "MULTIPLY"},
    )


def rounded_normal_inputs(nw: NodeWrangler, label):
    geometry = nw.new_node(Nodes.NewGeometry, label=f"{label}Geometry")
    rounded = nw.new_node(
        nodegroup_surface_normal(),
        input_kwargs={"Normal": geometry.outputs["Normal"]},
        label=label,
    )
    return {"Normal": rounded.outputs["Normal"], "Coat Normal": rounded.outputs["Normal"]}


def new_principled(nw: NodeWrangler, *, input_kwargs, attrs=None, label=None):
    kwargs = {"input_kwargs": input_kwargs}
    if attrs is not None:
        kwargs["attrs"] = attrs
    if label is not None:
        kwargs["label"] = label
    return nw.new_node(Nodes.PrincipledBSDF, **kwargs)


def assign_rgb_curves(node, points) -> None:
    for curve in node.mapping.curves[1:]:
        node_utils.assign_curve(curve, points)
    node.mapping.update()


@node_utils.to_nodegroup(SURFACE_NORMAL_GROUP_NAME)
def nodegroup_surface_normal(nw: NodeWrangler):
    size = EDGE_ROUNDING
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Normal", (0.0, 0.0, 0.0)),
            ("NodeSocketFloat", "Rough Surface Scale", 2.0),
        ],
    )
    geometry = nw.new_node(Nodes.NewGeometry)
    position = geometry.outputs["Position"]
    rough_surface = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": group_input.outputs["Rough Surface Scale"],
            "Detail": 0.0,
            "Distortion": 0.0,
        },
        label="Rough Surface",
    )
    bump = nw.new_node(
        "ShaderNodeBump",
        input_kwargs={
            "Strength": 0.2,
            "Distance": 0.005,
            "Height": rough_surface.outputs["Fac"],
            "Normal": group_input.outputs["Normal"],
        },
        label="Surface Bump",
    )
    normal = bump.outputs["Normal"]
    true_normal = geometry.outputs["True Normal"]
    incoming = geometry.outputs["Incoming"]

    def vmath(operation, a, b=None):
        inputs = {"Vector": a}
        if b is not None:
            inputs[1] = b
        return nw.new_node(Nodes.VectorMath, input_kwargs=inputs, attrs={"operation": operation})

    def scale(a, value):
        return nw.new_node(Nodes.VectorMath, input_kwargs={"Vector": a, "Scale": value}, attrs={"operation": "SCALE"})

    def fmath(operation, a, b=None):
        inputs = {0: a}
        if b is not None:
            inputs[1] = b
        return nw.new_node(Nodes.Math, input_kwargs=inputs, attrs={"operation": operation})

    def vmix(factor, a, b):
        return nw.new_node(
            Nodes.Mix,
            input_kwargs={"Factor": factor, "A": a, "B": b},
            attrs={"data_type": "VECTOR", "blend_type": "MIX"},
        )

    def dot(a, b):
        return nw.new_node(Nodes.VectorMath, input_kwargs={0: a, 1: b}, attrs={"operation": "DOT_PRODUCT"})

    ref1 = vmath("CROSS_PRODUCT", normal, (0.0, 0.0, 1.0))
    ref2 = vmath("CROSS_PRODUCT", normal, (1.0, 0.0, 0.0))
    separate = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": normal})
    nz = fmath("ABSOLUTE", separate.outputs["Z"])
    use_t2 = fmath("GREATER_THAN", nz.outputs["Value"], 0.9)
    tangent = vmath(
        "NORMALIZE", vmix(use_t2.outputs["Value"], ref1.outputs["Vector"], ref2.outputs["Vector"]).outputs["Result"]
    )
    bitangent = vmath("NORMALIZE", vmath("CROSS_PRODUCT", normal, tangent.outputs["Vector"]).outputs["Vector"])
    light_path = nw.new_node(Nodes.LightPath)
    through_depth = fmath(
        "ADD",
        light_path.outputs["Transmission Depth"],
        light_path.outputs["Transparent Depth"],
    )
    first_bounce = fmath("LESS_THAN", light_path.outputs["Ray Depth"], 2.0)
    through_only = fmath(
        "LESS_THAN",
        fmath(
            "ABSOLUTE",
            fmath("SUBTRACT", light_path.outputs["Ray Depth"], through_depth.outputs["Value"]).outputs["Value"],
        ).outputs["Value"],
        0.5,
    )
    shallow_through = fmath("LESS_THAN", through_depth.outputs["Value"], 3.0)
    through_bounce = fmath("MULTIPLY", through_only.outputs["Value"], shallow_through.outputs["Value"])
    rounding_gate = fmath("MAXIMUM", first_bounce.outputs["Value"], through_bounce.outputs["Value"])
    rounding_gate.name = rounding_gate.label = "Active Rounding"
    gated_length = fmath("MULTIPLY", rounding_gate.outputs["Value"], size)
    concave_position = vmath("ADD", position, scale(true_normal, 1e-5).outputs["Vector"])
    convex_position = vmath("ADD", position, scale(true_normal, -1e-5).outputs["Vector"])

    out_normal = normal
    decided = None
    for index in range(ROUNDING_EDGE_NORMAL_RAYCAST_SAMPLES):
        sample_u_key = vmath("ADD", position, (17.0 * index + 3.0, 29.0 * index + 5.0, 43.0 * index + 7.0))
        sample_v_key = vmath("ADD", position, (47.0 * index + 11.0, 53.0 * index + 13.0, 59.0 * index + 17.0))
        sample_u = nw.new_node(
            Nodes.WhiteNoiseTexture,
            input_kwargs={"Vector": sample_u_key.outputs["Vector"]},
            attrs={"noise_dimensions": "3D"},
        )
        sample_v = nw.new_node(
            Nodes.WhiteNoiseTexture,
            input_kwargs={"Vector": sample_v_key.outputs["Vector"]},
            attrs={"noise_dimensions": "3D"},
        )
        z = fmath(
            "DIVIDE",
            fmath("ADD", index, sample_u.outputs["Value"]).outputs["Value"],
            float(ROUNDING_EDGE_NORMAL_RAYCAST_SAMPLES),
        )
        radius = fmath(
            "SQRT",
            fmath(
                "MAXIMUM",
                fmath(
                    "SUBTRACT", 1.0, fmath("MULTIPLY", z.outputs["Value"], z.outputs["Value"]).outputs["Value"]
                ).outputs["Value"],
                0.0,
            ).outputs["Value"],
        )
        phi = fmath("MULTIPLY", sample_v.outputs["Value"], 2.0 * math.pi)
        cos_phi = fmath("COSINE", phi.outputs["Value"])
        sin_phi = fmath("SINE", phi.outputs["Value"])
        direction = vmath(
            "ADD",
            vmath(
                "ADD",
                scale(
                    tangent.outputs["Vector"],
                    fmath("MULTIPLY", radius.outputs["Value"], cos_phi.outputs["Value"]).outputs["Value"],
                ).outputs["Vector"],
                scale(
                    bitangent.outputs["Vector"],
                    fmath("MULTIPLY", radius.outputs["Value"], sin_phi.outputs["Value"]).outputs["Value"],
                ).outputs["Vector"],
            ).outputs["Vector"],
            scale(normal, z.outputs["Value"]).outputs["Vector"],
        )
        direction_normalized = vmath("NORMALIZE", direction.outputs["Vector"])
        negative_direction = scale(direction_normalized.outputs["Vector"], -1.0)

        concave = nw.new_node(
            "ShaderNodeRaycast",
            input_kwargs={
                "Position": concave_position.outputs["Vector"],
                "Direction": direction_normalized.outputs["Vector"],
                "Length": gated_length.outputs["Value"],
            },
            # Object-local raycast: a scene-wide raycast hits the shadow-catcher
            # ground under bottom rims, corrupting rounded normals.
            attrs={"only_local": True},
        )
        convex = nw.new_node(
            "ShaderNodeRaycast",
            input_kwargs={
                "Position": convex_position.outputs["Vector"],
                "Direction": negative_direction.outputs["Vector"],
                "Length": gated_length.outputs["Value"],
            },
            attrs={"only_local": True},
        )

        concave_weight = fmath(
            "MAXIMUM",
            fmath(
                "SUBTRACT",
                1.0,
                fmath("DIVIDE", concave.outputs["Hit Distance"], size).outputs["Value"],
            ).outputs["Value"],
            0.0,
        )
        concave_normal = vmath(
            "ADD",
            normal,
            scale(concave.outputs["Hit Normal"], concave_weight.outputs["Value"]).outputs["Vector"],
        )
        concave_gate = fmath("GREATER_THAN", dot(incoming, concave.outputs["Hit Normal"]).outputs["Value"], 0.0)
        valid_concave = fmath("MULTIPLY", concave.outputs["Is Hit"], concave_gate.outputs["Value"])

        convex_weight = fmath(
            "MAXIMUM",
            fmath(
                "SUBTRACT",
                1.0,
                fmath("DIVIDE", convex.outputs["Hit Distance"], size).outputs["Value"],
            ).outputs["Value"],
            0.0,
        )
        convex_hit = scale(convex.outputs["Hit Normal"], -1.0)
        convex_normal = vmath(
            "ADD",
            normal,
            scale(convex_hit.outputs["Vector"], convex_weight.outputs["Value"]).outputs["Vector"],
        )
        convex_gate = fmath("GREATER_THAN", dot(incoming, convex_hit.outputs["Vector"]).outputs["Value"], 0.0)
        valid_convex = fmath("MULTIPLY", convex.outputs["Is Hit"], convex_gate.outputs["Value"])

        tangent_raw = vmath(
            "SUBTRACT",
            convex_hit.outputs["Vector"],
            scale(normal, dot(convex_hit.outputs["Vector"], normal).outputs["Value"]).outputs["Vector"],
        )
        tangent_dir = vmath("NORMALIZE", tangent_raw.outputs["Vector"])
        probe_origin = vmath(
            "ADD",
            vmath("ADD", position, scale(tangent_dir.outputs["Vector"], size).outputs["Vector"]).outputs["Vector"],
            scale(normal, size * 0.5).outputs["Vector"],
        )
        probe = nw.new_node(
            "ShaderNodeRaycast",
            input_kwargs={
                "Position": probe_origin.outputs["Vector"],
                "Direction": scale(normal, -1.0).outputs["Vector"],
                "Length": gated_length.outputs["Value"],
            },
            attrs={"only_local": True},
        )
        surface_ends = fmath("SUBTRACT", 1.0, probe.outputs["Is Hit"])
        valid_convex = fmath("MULTIPLY", valid_convex.outputs["Value"], surface_ends.outputs["Value"])

        concave_candidate = vmix(valid_concave.outputs["Value"], out_normal, concave_normal.outputs["Vector"])
        concave_missed = fmath("SUBTRACT", 1.0, concave.outputs["Is Hit"])
        convex_allowed = fmath("MULTIPLY", concave_missed.outputs["Value"], valid_convex.outputs["Value"])
        candidate = vmix(
            convex_allowed.outputs["Value"], concave_candidate.outputs["Result"], convex_normal.outputs["Vector"]
        )
        hit_any = fmath("MAXIMUM", concave.outputs["Is Hit"], convex.outputs["Is Hit"])
        if decided is None:
            out_normal = vmix(hit_any.outputs["Value"], out_normal, candidate.outputs["Result"]).outputs["Result"]
            decided = hit_any.outputs["Value"]
        else:
            take = fmath("MULTIPLY", fmath("SUBTRACT", 1.0, decided).outputs["Value"], hit_any.outputs["Value"])
            out_normal = vmix(take.outputs["Value"], out_normal, candidate.outputs["Result"]).outputs["Result"]
            if index != ROUNDING_EDGE_NORMAL_RAYCAST_SAMPLES - 1:
                decided = fmath("MAXIMUM", decided, hit_any.outputs["Value"]).outputs["Value"]

    rounded = vmath("NORMALIZE", out_normal)
    camera_only = vmix(rounding_gate.outputs["Value"], normal, rounded.outputs["Vector"])
    camera_only.name = camera_only.label = "Rounded Normal or Passthrough"
    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Normal": camera_only.outputs["Result"]},
        label="GroupOutput",
    )


def _build_trans_neon_group(nw: NodeWrangler, base_group, include_chip_color=False):
    exposed_inputs = [
        ("NodeSocketColor", "WhiteColor", (0.0000, 0.0000, 0.0000, 1.0000)),
        ("NodeSocketColor", "Color", (0.0000, 0.0000, 0.0000, 1.0000)),
    ]
    if include_chip_color:
        exposed_inputs.append(("NodeSocketColor", "ChipColor", (0.0000, 0.0000, 0.0000, 1.0000)))
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=exposed_inputs,
    )

    base_inputs = {"WhiteColor": group_input.outputs["WhiteColor"], "Color": group_input.outputs["Color"]}
    if include_chip_color:
        base_inputs["ChipColor"] = group_input.outputs["ChipColor"]
    base = nw.new_node(base_group(), input_kwargs=base_inputs, label="TRANS-GROUP_BASE")

    layer_weight = nw.new_node(
        Nodes.LayerWeight,
        input_kwargs={"Blend": 0.2000, "Normal": base.outputs["Normal"]},
        label="LayerWeight",
    )

    translucent_bsdf = nw.new_node(
        Nodes.TranslucentBSDF,
        input_kwargs={
            "Color": group_input.outputs["Color"],
            **({} if include_chip_color else {"Normal": base.outputs["Normal"]}),
        },
        label="TransNeonGroupTranslucentBSDF",
    )

    trans_neon_group_add_shader = nw.new_node(
        "ShaderNodeAddShader",
        input_kwargs={0: base.outputs["Shader"], 1: translucent_bsdf},
        label="TransNeonGroupAddShader",
    )

    trans_neon_group_mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Factor": layer_weight.outputs["Fresnel"],
            1: base.outputs["Shader"],
            2: trans_neon_group_add_shader,
        },
        label="TransNeonGroupMixShader",
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Shader": trans_neon_group_mix_shader, "Volume": base.outputs["Volume"]},
        label="GroupOutput",
    )


@node_utils.to_nodegroup("Mat_Transparent Neon")
def nodegroup_trans_neon_group(nw: NodeWrangler):
    _build_trans_neon_group(nw, nodegroup_trans_group_base, False)


@node_utils.to_nodegroup("Mat_Transparent")
def nodegroup_trans_group_base(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "WhiteColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketColor", "Color", (0.0000, 0.0000, 0.0000, 1.0000)),
        ],
    )
    normal_group = _rounded_normal(nw, "TransGroupBaseNormalGroup")

    trans_group_base_volume_absorption = nw.new_node(
        "ShaderNodeVolumeAbsorption",
        input_kwargs={"Color": group_input.outputs["Color"], "Density": 6.0000},
        label="TransGroupBaseVolumeAbsorption",
    )

    glass_shader = fresnel_glass_mix(
        nw,
        group_input.outputs["WhiteColor"],
        0.05000000074505806,
        normal_group,
        "TransGroupBase",
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Shader": glass_shader.outputs["Shader"],
            "Volume": trans_group_base_volume_absorption,
            "Normal": normal_group,
        },
        label="GroupOutput",
    )


@node_utils.to_nodegroup("Mat_LDraw Particle Distribution")
def nodegroup_ldraw_particle_distribution(nw: NodeWrangler):
    """Build an object-space particle mask at three sizes within LDraw bounds."""

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVector", "Position", (0.0, 0.0, 0.0)),
            ("NodeSocketFloat", "Minimum Size", 1.0),
            ("NodeSocketFloat", "Maximum Size", 1.0),
            ("NodeSocketFloat", "Fraction", 0.0),
        ],
    )
    small_scale = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["Minimum Size"]},
        label="Small Particle Scale",
        attrs={"operation": "DIVIDE"},
    )
    medium_size = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            "Factor": 0.5,
            "A": group_input.outputs["Minimum Size"],
            "B": group_input.outputs["Maximum Size"],
        },
        label="Medium Particle Size",
        attrs={"data_type": "FLOAT"},
    )
    medium_scale = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0, 1: medium_size.outputs["Result"]},
        label="Medium Particle Scale",
        attrs={"operation": "DIVIDE"},
    )
    large_scale = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0, 1: group_input.outputs["Maximum Size"]},
        label="Large Particle Scale",
        attrs={"operation": "DIVIDE"},
    )

    def cell_mask(label, scale):
        cells = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={"Vector": group_input.outputs["Position"], "Scale": scale},
            label="%s Particle Cells" % label,
        )
        c = nw.new_node(
            Nodes.SeparateColor,
            input_kwargs={"Color": cells.outputs["Color"]},
            label="%s Particle Random" % label,
        )
        mask = nw.new_node(
            Nodes.Math,
            input_kwargs={0: c.outputs["Red"], 1: group_input.outputs["Fraction"]},
            label="%s Particle Mask" % label,
            attrs={"operation": "LESS_THAN"},
        )
        return mask.outputs["Value"], c.outputs["Green"]

    small_mask, _ = cell_mask("Small", small_scale)
    medium_mask, _ = cell_mask("Medium", medium_scale)
    large_mask, size_random = cell_mask("Large", large_scale)
    large_region = nw.new_node(
        Nodes.Math,
        input_kwargs={0: size_random, 1: 2.0 / 3.0},
        label="Large Particle Region",
        attrs={"operation": "GREATER_THAN"},
    )
    medium_or_large = nw.new_node(
        Nodes.Mix,
        input_kwargs={"Factor": large_region, "A": medium_mask, "B": large_mask},
        label="Medium or Large Particles",
        attrs={"data_type": "FLOAT"},
    )
    small_region = nw.new_node(
        Nodes.Math,
        input_kwargs={0: size_random, 1: 1.0 / 3.0},
        label="Small Particle Region",
        attrs={"operation": "LESS_THAN"},
    )
    mask = nw.new_node(
        Nodes.Mix,
        input_kwargs={"Factor": small_region, "A": medium_or_large.outputs["Result"], "B": small_mask},
        label="Particle Mask",
        attrs={"data_type": "FLOAT"},
    )
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Mask": mask.outputs["Result"]})


@node_utils.to_nodegroup("Mat_Speckle")
def nodegroup_speckle_group(nw: NodeWrangler):
    speckle_group_geometry = nw.new_node(Nodes.NewGeometry, label="SpeckleGroupGeometry")

    speckle_noise = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": speckle_group_geometry.outputs["Position"], "Scale": 50.0000},
        label="SpeckleNoise",
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketColor", "ChipColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "XOffset", 0.0000),
            ("NodeSocketFloat", "Particle Scale", 50.0000),
            ("NodeSocketFloat", "Particle Detail", 2.0000),
            ("NodeSocketFloat", "Minimum Particle Size", 1.0000),
            ("NodeSocketFloat", "Maximum Particle Size", 1.0000),
            ("NodeSocketFloat", "Chip Fraction", 0.0000),
            ("NodeSocketFloat", "LDraw Distribution", 0.0000),
            ("NodeSocketFloat", "Chip Alpha", 1.0000),
            ("NodeSocketFloat", "Chip Luminance", 0.0000),
        ],
    )
    speckle_noise.inputs["Scale"].default_value = 0.0
    nw.links.new(group_input.outputs["Particle Scale"], speckle_noise.inputs["Scale"])
    speckle_noise.inputs["Detail"].default_value = 0.0
    nw.links.new(group_input.outputs["Particle Detail"], speckle_noise.inputs["Detail"])

    speckle_group_sigmoid_group = nw.new_node(
        nodegroup_sigmoidgroup(),
        input_kwargs={
            "X": speckle_noise.outputs["Color"],
            "XOffset": group_input.outputs["XOffset"],
        },
        label="SpeckleGroupSigmoidGroup",
    )
    texture_coordinate = nw.new_node(Nodes.TextureCoord, label="LDraw Speckle Coordinates")
    custom_speckle = nw.new_node(
        nodegroup_ldraw_particle_distribution(),
        input_kwargs={
            "Position": texture_coordinate.outputs["Object"],
            "Minimum Size": group_input.outputs["Minimum Particle Size"],
            "Maximum Size": group_input.outputs["Maximum Particle Size"],
            "Fraction": group_input.outputs["Chip Fraction"],
        },
        label="LDraw Speckle Distribution",
    )
    speckle_weight = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            "Factor": group_input.outputs["LDraw Distribution"],
            "A": speckle_group_sigmoid_group,
            "B": custom_speckle.outputs["Mask"],
        },
        label="Speckle Weight",
        attrs={"data_type": "FLOAT"},
    )

    speckle_rounded = rounded_normal_inputs(nw, "SpeckleGroupRoundingEdgeNormal")

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["BaseColor"],
            "Roughness": 0.3000,
            "IOR": 1.4900,
            **speckle_rounded,
        },
        label="SpeckleGroupPrincipledBSDF",
        attrs=GGX_BURLEY,
    )

    principled_bsdf_1 = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["ChipColor"],
            "Metallic": 1.0000,
            "Roughness": 0.3000,
            "Alpha": group_input.outputs["Chip Alpha"],
            "Emission Color": group_input.outputs["ChipColor"],
            "Emission Strength": group_input.outputs["Chip Luminance"],
            **speckle_rounded,
        },
        attrs=GGX_BURLEY,
    )

    speckle_group_mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": speckle_weight.outputs["Result"], 1: principled_bsdf, 2: principled_bsdf_1},
        label="SpeckleGroupMixShader",
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"Shader": speckle_group_mix_shader}, label="GroupOutput")


BURLEY_S0 = 0.9501
BURLEY_FD90 = 1.6443
BURLEY_SHEEN_LOBES = ((0.3835, 0.3), (0.0499, 0.8))  # (strength k, sheen roughness)
SPECULAR_IOR = 1.5
PEARL_SUBSURFACE_SCALE = 0.65
SATIN_SUBSURFACE_SCALE = 0.65
MILKY_SUBSURFACE_SCALE = 0.85
PEARL_COAT_FACTOR = 0.25
PEARL_BURLEY_S0 = 0.9536
PEARL_BURLEY_SHEEN_LOBES = ((0.4593, 0.8),)


def burley_sss_radius_fitting(nw: NodeWrangler, base_color_socket, radius_socket, label_prefix):
    """Build the per-channel Burley radius fit."""
    separate = nw.new_node(
        "ShaderNodeSeparateColor", input_kwargs={"Color": base_color_socket}, label=f"{label_prefix}SeparateA"
    )
    inv_channels = []
    for c in ("Red", "Green", "Blue"):
        slope = nw.new_node(
            Nodes.Math,
            input_kwargs={0: separate.outputs[c], 1: 3.5000, 2: -6.6000},
            label=f"{label_prefix}Fit1{c}",
            attrs={"operation": "MULTIPLY_ADD"},
        )
        s_value = nw.new_node(
            Nodes.Math,
            input_kwargs={0: slope, 1: separate.outputs[c], 2: 4.1400},
            label=f"{label_prefix}Fit2{c}",
            attrs={"operation": "MULTIPLY_ADD"},
        )
        inv_channels.append(
            nw.new_node(
                Nodes.Math,
                input_kwargs={0: 1.0000, 1: s_value},
                label=f"{label_prefix}InvS{c}",
                attrs={"operation": "DIVIDE"},
            )
        )
    inv_s = nw.new_node(
        "ShaderNodeCombineXYZ",
        input_kwargs={"X": inv_channels[0], "Y": inv_channels[1], "Z": inv_channels[2]},
        label=f"{label_prefix}InvSVector",
    )
    if isinstance(radius_socket, (tuple, list)) and tuple(radius_socket) == (1.0, 1.0, 1.0):
        return inv_s
    return nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: radius_socket, 1: inv_s},
        label=f"{label_prefix}FittedRadius",
        attrs={"operation": "MULTIPLY"},
    )


def fresnel_glass_mix(nw: NodeWrangler, color_socket, roughness_socket, normal_socket, label_prefix):
    """Mix tinted refraction and white reflection by Fresnel weight."""
    glass_fresnel = nw.new_node(
        Nodes.Fresnel,
        input_kwargs={"IOR": 1.4900, "Normal": normal_socket},
        label=f"{label_prefix}GlassFresnel",
    )
    refraction_bsdf = nw.new_node(
        Nodes.RefractionBSDF,
        input_kwargs={
            "Color": color_socket,
            "Roughness": roughness_socket,
            "IOR": 1.4900,
            "Normal": normal_socket,
        },
        label=f"{label_prefix}GlassRefraction",
        attrs=_GGX,
    )
    glossy_bsdf = nw.new_node(
        Nodes.GlossyBSDF,
        input_kwargs={
            "Color": (1.0000, 1.0000, 1.0000, 1.0000),
            "Roughness": roughness_socket,
            "Normal": normal_socket,
        },
        label=f"{label_prefix}GlassReflection",
        attrs=_GGX,
    )
    return nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": glass_fresnel.outputs["Fac"], 1: refraction_bsdf, 2: glossy_bsdf},
        label=f"{label_prefix}GlassShader",
    )


def burley_view_factor(nw: NodeWrangler, normal_socket, label_prefix):
    """F_v = S0 * (1 + (FD90 - 1) * (1 - N.V)^5) as a scalar Math-node chain."""
    geometry = nw.new_node(Nodes.NewGeometry, label=f"{label_prefix}Geometry")
    n_dot_v = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: normal_socket, 1: geometry.outputs["Incoming"]},
        label=f"{label_prefix}NdotV",
        attrs={"operation": "DOT_PRODUCT"},
    )
    one_minus = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: n_dot_v.outputs["Value"]},
        label=f"{label_prefix}OneMinus",
        attrs={"operation": "SUBTRACT"},
    )
    pow5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: one_minus, 1: 5.0000},
        label=f"{label_prefix}Pow5",
        attrs={"operation": "POWER"},
    )
    fresnel = nw.new_node(
        Nodes.Math,
        input_kwargs={0: pow5, 1: BURLEY_FD90 - 1.0, 2: 1.0000},
        label=f"{label_prefix}Fresnel",
        attrs={"operation": "MULTIPLY_ADD"},
    )
    return nw.new_node(
        Nodes.Math,
        input_kwargs={0: fresnel, 1: BURLEY_S0},
        label=f"{label_prefix}Scale",
        attrs={"operation": "MULTIPLY"},
    )


def burley_sheen_lobes(nw: NodeWrangler, shader, color_socket, normal_socket, label_prefix, lobes=None):
    """Chain AddShader'ed Sheen BSDF lobes (tinted color * k) onto `shader`."""
    for k, roughness in BURLEY_SHEEN_LOBES if lobes is None else lobes:
        tint = _color_multiply(nw, color_socket, (k, k, k, 1.0), f"{label_prefix}SheenTint{roughness}")
        sheen = nw.new_node(
            "ShaderNodeBsdfSheen",
            input_kwargs={"Color": tint, "Roughness": roughness, "Normal": normal_socket},
            label=f"{label_prefix}Sheen{roughness}",
        )
        shader = nw.new_node(
            "ShaderNodeAddShader",
            input_kwargs={0: shader, 1: sheen},
            label=f"{label_prefix}SheenAdd{roughness}",
        )
    return shader


@node_utils.to_nodegroup("Mat_Solid")
def nodegroup_solid_group(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000)),
        ],
    )

    solid_group_rounding_edge_normal = rounded_normal_inputs(nw, "SolidGroupRoundingEdgeNormal")

    view_factor = burley_view_factor(nw, solid_group_rounding_edge_normal["Normal"], "SolidGroupBurley")
    burley_base_color = _color_multiply(nw, group_input.outputs["BaseColor"], view_factor, "SolidGroupBurleyBaseColor")

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": burley_base_color,
            "Roughness": 0.0500,
            "IOR": SPECULAR_IOR,
            "Normal": solid_group_rounding_edge_normal["Normal"],
            "Coat Weight": 0.2500,
            "Coat Roughness": 0.0000,
            "Coat Normal": solid_group_rounding_edge_normal["Coat Normal"],
        },
        attrs=GGX_BURLEY,
    )

    shader = burley_sheen_lobes(
        nw,
        principled_bsdf,
        group_input.outputs["BaseColor"],
        solid_group_rounding_edge_normal["Normal"],
        "SolidGroupBurley",
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"Shader": shader}, label="GroupOutput")


@node_utils.to_nodegroup("Mat_Sigmoid")
def nodegroup_sigmoidgroup(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketFloat", "X", 0.0000),
            ("NodeSocketFloat", "XOffset", 0.0000),
        ],
    )

    sigmoid_group_subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["XOffset"], 1: group_input.outputs["X"]},
        label="SigmoidGroupSubtract",
        attrs={"operation": "SUBTRACT"},
    )

    sigmoid_group_multiply2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: sigmoid_group_subtract, 1: 100.0000},
        label="SigmoidGroupMultiply2",
        attrs={"operation": "MULTIPLY"},
    )

    sigmoid_group_power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 2.7128, 1: sigmoid_group_multiply2},
        label="SigmoidGroupPower",
        attrs={"operation": "POWER"},
    )

    sigmoid_group_add = nw.new_node(
        Nodes.Math, input_kwargs={0: 1.0000, 1: sigmoid_group_power}, label="SigmoidGroupAdd"
    )

    sigmoid_group_divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: sigmoid_group_add},
        label="SigmoidGroupDivide",
        attrs={"operation": "DIVIDE"},
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Out": sigmoid_group_divide},
        label="GroupOutput",
    )


@node_utils.to_nodegroup("Mat_Shine")
def nodegroup_shine_group(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "ShineColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketVector", "SubsurfaceRadius", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "SubsurfaceRoughness", 0.0000),
            ("NodeSocketFloat", "Metallic", 0.0000),
            ("NodeSocketFloat", "LayerWeight", 0.0000),
        ],
    )

    layer_weight = nw.new_node(
        Nodes.LayerWeight, input_kwargs={"Blend": group_input.outputs["LayerWeight"]}, label="LayerWeight"
    )

    mix = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            0: layer_weight.outputs["Facing"],
            6: group_input.outputs["ShineColor"],
            7: group_input.outputs["BaseColor"],
        },
        attrs={"data_type": "RGBA"},
    )

    principled_group = nw.new_node(
        nodegroup_principled_group(),
        input_kwargs={
            "BaseColor": mix.outputs["Result"],
            "SubsurfaceRoughness": group_input.outputs["SubsurfaceRoughness"],
            "Clearcoat": 0.3000,
            "SubsurfaceRadius": group_input.outputs["SubsurfaceRadius"],
            "Metallic": group_input.outputs["Metallic"],
        },
        label="PRINCIPLED-GROUP",
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"BSDF": principled_group},
        label="GroupOutput",
    )


@node_utils.to_nodegroup("Mat_Satin Transparent")
def nodegroup_satin_trans_group(nw: NodeWrangler):
    _build_satin_group(nw, "SatinTransGroup", 0.9000)


def _build_satin_group(nw: NodeWrangler, label_prefix, transmission_weight):
    satin_group_geometry = nw.new_node(Nodes.NewGeometry, label=f"{label_prefix}Geometry")

    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "WhiteColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketColor", "Color", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "VoronoiTextureScale", 0.0000),
        ],
    )

    satin_group_voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": satin_group_geometry.outputs["Position"],
            "Scale": group_input.outputs["VoronoiTextureScale"],
        },
        label=f"{label_prefix}VoronoiTexture",
    )

    satin_rgb_curves = nw.new_node(
        Nodes.RGBCurve,
        input_kwargs={"Color": satin_group_voronoi_texture.outputs["Color"]},
        label=f"{label_prefix}RGBCurves",
    )
    assign_rgb_curves(satin_rgb_curves, curve_data.SATIN_RGB_CURVE)

    normals = rounded_normal_inputs(nw, f"{label_prefix}RoundingEdgeNormal")

    fitted_radius = burley_sss_radius_fitting(
        nw, group_input.outputs["Color"], (1.0, 1.0, 1.0), f"{label_prefix}Burley"
    )

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["Color"],
            "Roughness": 0.0000,
            "IOR": SPECULAR_IOR,
            **normals,
            "Subsurface Weight": 1.0000,
            "Subsurface Radius": fitted_radius,
            "Subsurface Scale": SATIN_SUBSURFACE_SCALE,
        },
        label=f"{label_prefix}PrincipledBSDF",
        attrs=GGX_BURLEY,
    )

    glass_shader = fresnel_glass_mix(nw, group_input.outputs["Color"], 0.0000, normals["Normal"], label_prefix)
    trans_mix = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": transmission_weight, 1: principled_bsdf, 2: glass_shader},
        label=f"{label_prefix}GlassMix",
    )

    principled_bsdf_1 = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["WhiteColor"],
            "Metallic": 0.5000,
            "Roughness": 0.0000,
            "IOR": 1.4900,
            **normals,
        },
        label=f"{label_prefix}GlossyBSDF",
        attrs=_GGX,
    )

    satin_group_mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": satin_rgb_curves, 1: trans_mix, 2: principled_bsdf_1},
        label=f"{label_prefix}MixShader",
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"BSDF": satin_group_mix_shader}, label="GroupOutput")


@node_utils.to_nodegroup("Mat_Satin")
def nodegroup_satin_group(nw: NodeWrangler):
    _build_satin_group(nw, "SatinGroup", 0.6500)


@node_utils.to_nodegroup("Mat_Rubber Transparent")
def nodegroup_rubber_trans_group(nw: NodeWrangler):
    _build_rubber_group(nw, 0.9500, label_prefix="RubberTransGroup")


RUBBER_TRANS_GLASS_IOR = 1.49
RUBBER_TRANS_GLOSS_LOSS_QUAD = (0.04613, -0.03460, 0.23823)
RUBBER_TRANS_REFR_LOSS_QUAD = (0.0, 0.0, 0.0278)


def _rubber_trans_loss_dim(nw: NodeWrangler, c, coeffs, tag):
    """Return the roughness-dependent transmission multiplier."""
    a, b, cc = coeffs
    r2 = nw.new_node(Nodes.Math, input_kwargs={0: c, 1: c}, label=f"{tag}R2", attrs={"operation": "MULTIPLY"})
    cr2 = nw.new_node(Nodes.Math, input_kwargs={0: r2, 1: cc}, label=f"{tag}C", attrs={"operation": "MULTIPLY"})
    loss = cr2
    if b != 0:
        br = nw.new_node(Nodes.Math, input_kwargs={0: c, 1: b}, label=f"{tag}B", attrs={"operation": "MULTIPLY"})
        loss = nw.new_node(Nodes.Math, input_kwargs={0: br, 1: cr2}, label=f"{tag}S1", attrs={"operation": "ADD"})
    if a != 0:
        loss = nw.new_node(
            Nodes.Math,
            input_kwargs={0: loss, 1: a},
            label=f"{tag}Loss",
            attrs={"operation": "ADD", "use_clamp": True},
        )
    else:
        loss.use_clamp = True
    return nw.new_node(Nodes.Math, input_kwargs={0: 1.0, 1: loss}, label=f"{tag}Dim", attrs={"operation": "SUBTRACT"})


def _build_rubber_group(nw: NodeWrangler, transmission_weight, label_prefix="RubberTransGroup"):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000)),
        ],
    )

    base_color = group_input.outputs["BaseColor"]
    roughness = base_color

    normal_inputs = rounded_normal_inputs(nw, f"{label_prefix}RoundingEdgeNormal")
    normal = normal_inputs["Normal"]

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": base_color,
            "Roughness": roughness,
            "IOR": SPECULAR_IOR,
            **normal_inputs,
        },
        label=f"{label_prefix}PrincipledBSDF",
        attrs=GGX_BURLEY,
    )

    refr_color = _color_multiply(
        nw,
        base_color,
        _rubber_trans_loss_dim(nw, base_color, RUBBER_TRANS_REFR_LOSS_QUAD, f"{label_prefix}TransDim"),
        f"{label_prefix}RefrTint",
    )
    refraction_bsdf = nw.new_node(
        Nodes.RefractionBSDF,
        input_kwargs={
            "Color": refr_color,
            "Roughness": roughness,
            "IOR": RUBBER_TRANS_GLASS_IOR,
            "Normal": normal,
        },
        label=f"{label_prefix}RefractionBSDF",
        attrs=_GGX,
    )
    glossy_bsdf = nw.new_node(
        Nodes.GlossyBSDF,
        input_kwargs={
            "Color": _rubber_trans_loss_dim(nw, base_color, RUBBER_TRANS_GLOSS_LOSS_QUAD, f"{label_prefix}GlossDim"),
            "Roughness": roughness,
            "Normal": normal,
        },
        label=f"{label_prefix}GlossyBSDF",
        attrs=_GGX,
    )
    fresnel = nw.new_node(
        Nodes.Fresnel,
        input_kwargs={"IOR": RUBBER_TRANS_GLASS_IOR, "Normal": normal},
        label=f"{label_prefix}Fresnel",
    )
    glass_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": fresnel.outputs["Fac"], 1: refraction_bsdf, 2: glossy_bsdf},
        label=f"{label_prefix}GlassShader",
    )
    shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": transmission_weight, 1: principled_bsdf, 2: glass_shader},
        label=f"{label_prefix}GlassMix",
    )
    nw.new_node(Nodes.GroupOutput, input_kwargs={"BSDF": shader}, label="GroupOutput")


@node_utils.to_nodegroup("Mat_Rubber")
def nodegroup_rubber_group(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000)),
        ],
    )
    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["BaseColor"],
            "Roughness": 0.7000,
            "IOR": 1.4900,
            **rounded_normal_inputs(nw, "RubberGroupRoundingEdgeNormal"),
        },
        label="RubberGroupPrincipledBSDF",
        attrs=GGX_BURLEY,
    )
    nw.new_node(Nodes.GroupOutput, input_kwargs={"BSDF": principled_bsdf}, label="GroupOutput")


@node_utils.to_nodegroup("Mat_Principled")
def nodegroup_principled_group(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketVector", "SubsurfaceRadius", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "SubsurfaceRoughness", 0.0000),
            ("NodeSocketFloat", "Clearcoat", 0.0000),
            ("NodeSocketFloat", "Metallic", 0.0000),
        ],
    )

    normals = rounded_normal_inputs(nw, "PearlGroupRoundingEdgeNormal")

    burley_base_color = _color_multiply(
        nw,
        group_input.outputs["BaseColor"],
        (PEARL_BURLEY_S0, PEARL_BURLEY_S0, PEARL_BURLEY_S0, 1.0),
        "PearlGroupBurleyBaseColor",
    )

    coat_weight = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Clearcoat"], 1: PEARL_COAT_FACTOR},
        label="PearlGroupCoatWeight",
        attrs={"operation": "MULTIPLY"},
    )

    fitted_radius = burley_sss_radius_fitting(
        nw, group_input.outputs["BaseColor"], group_input.outputs["SubsurfaceRadius"], "PearlGroupBurley"
    )

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": burley_base_color,
            "Metallic": group_input.outputs["Metallic"],
            "Roughness": group_input.outputs["SubsurfaceRoughness"],
            "IOR": SPECULAR_IOR,
            **normals,
            "Subsurface Weight": 1.0000,
            "Subsurface Radius": fitted_radius,
            "Subsurface Scale": PEARL_SUBSURFACE_SCALE,
            "Specular IOR Level": 0.7000,
            "Coat Weight": coat_weight,
            "Coat Roughness": 0.0000,
        },
        attrs=GGX_BURLEY,
    )

    shader = burley_sheen_lobes(
        nw,
        principled_bsdf,
        group_input.outputs["BaseColor"],
        normals["Normal"],
        "PearlGroupBurley",
        lobes=PEARL_BURLEY_SHEEN_LOBES,
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"BSDF": shader}, label="GroupOutput")


def _build_pearl_group(nw: NodeWrangler, roughness_value, glossy_roughness):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "Color", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Subsurface", 0.0000),
            ("NodeSocketFloat", "Clearcoat", 0.0000),
            ("NodeSocketFloat", "Metallic", 0.0000),
        ],
    )

    normal_group = _rounded_normal(nw, "PearlGroupNormalGroup")

    subsurface_radius = burley_sss_radius_fitting(
        nw, group_input.outputs["Color"], (1.0, 1.0, 1.0), "PearlGroupSubsurfaceRadius"
    )

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["Color"],
            "Metallic": group_input.outputs["Metallic"],
            "Roughness": roughness_value,
            "IOR": 1.4900,
            "Normal": normal_group,
            "Subsurface Weight": 1.0000,
            "Subsurface Radius": subsurface_radius,
            "Subsurface Scale": group_input.outputs["Subsurface"],
            "Coat Weight": group_input.outputs["Clearcoat"],
            "Coat Roughness": 0.0000,
        },
        label="PearlGroupPrincipledBSDF",
        attrs=GGX_BURLEY,
    )

    pearl_glossy_bsdf = nw.new_node(
        Nodes.GlossyBSDF,
        input_kwargs={
            "Roughness": glossy_roughness,
            "Normal": normal_group,
        },
        label="PearlGroupGlossyBSDF",
        attrs=_GGX,
    )

    pearl_group_mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": 0.1000, 1: principled_bsdf, 2: pearl_glossy_bsdf},
        label="PearlGroupMixShader",
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"BSDF": pearl_group_mix_shader.outputs["Shader"]},
        label="GroupOutput",
    )


@node_utils.to_nodegroup("Mat_Pearl")
def nodegroup_pearl_group(nw: NodeWrangler):
    _build_pearl_group(nw, 0.2500, 0.0000)


@node_utils.to_nodegroup("Mat_Pearl Flat")
def nodegroup_pearl_flat_group(nw: NodeWrangler):
    _build_pearl_group(nw, 0.5000, 0.3000)


@node_utils.to_nodegroup("Mat_Metal")
def nodegroup_metal_group(nw: NodeWrangler):
    coord_fix = nw.new_node(Nodes.TextureCoord, label="Metal Texture Coordinates")
    metal_group_noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": coord_fix.outputs["Object"], "Scale": 260.0000, "Detail": 0.0000},
        label="MetalGroupNoiseTexture",
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000)),
        ],
    )

    metal_normal = _rounded_normal(nw, "MetalGroupRoundingEdgeNormal")

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["BaseColor"],
            "Metallic": 1.0000,
            "Roughness": 0.3500,
            "IOR": 1.4500,
            "Normal": metal_normal,
        },
        attrs=_GGX,
    )

    metal_glossy_bsdf = nw.new_node(
        "ShaderNodeBsdfAnisotropic",
        input_kwargs={
            "Color": group_input.outputs["BaseColor"],
            "Roughness": 0.3500,
            "Normal": metal_normal,
        },
        label="MetalGroupGlossyBSDF",
        attrs=_GGX,
    )

    metal_group_mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Factor": metal_group_noise_texture.outputs["Factor"],
            1: principled_bsdf,
            2: metal_glossy_bsdf,
        },
        label="MetalGroupMixShader",
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"Shader": metal_group_mix_shader}, label="GroupOutput")


@node_utils.to_nodegroup("Mat_Glitter Transparent Neon")
def nodegroup_glitter_trans_neon_group(nw: NodeWrangler):
    _build_trans_neon_group(nw, nodegroup_glitter_trans_group, True)


@node_utils.to_nodegroup("Mat_Glitter Transparent")
def nodegroup_glitter_trans_group(nw: NodeWrangler):
    glitter_trans_group_normal_group = _rounded_normal(nw, "GlitterTransGroupNormalGroup")

    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[
            ("NodeSocketColor", "WhiteColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketColor", "Color", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketColor", "ChipColor", (0.0000, 0.0000, 0.0000, 1.0000)),
            ("NodeSocketFloat", "Particle Scale", 30.0000),
            ("NodeSocketFloat", "Minimum Particle Size", 1.0000),
            ("NodeSocketFloat", "Maximum Particle Size", 1.0000),
            ("NodeSocketFloat", "Chip Fraction", 0.0000),
            ("NodeSocketFloat", "LDraw Distribution", 0.0000),
            ("NodeSocketFloat", "Chip Alpha", 1.0000),
            ("NodeSocketFloat", "Chip Luminance", 0.0000),
        ],
    )

    glitter_trans_group_volume_absorption = nw.new_node(
        "ShaderNodeVolumeAbsorption",
        input_kwargs={"Color": group_input.outputs["Color"], "Density": 6.0000},
        label="GlitterTransGroupVolumeAbsorption",
    )

    glitter_trans_group_geometry = nw.new_node(Nodes.NewGeometry, label="GlitterTransGroupGeometry")

    glitter_trans_group_voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            "Vector": glitter_trans_group_geometry.outputs["Position"],
            "Scale": group_input.outputs["Particle Scale"],
        },
        label="GlitterTransGroupVoronoiTexture",
    )

    glitter_rgb_curves = nw.new_node(
        Nodes.RGBCurve,
        input_kwargs={"Color": glitter_trans_group_voronoi_texture.outputs["Color"]},
        label="GlitterTransGroupRGBCurves",
    )
    assign_rgb_curves(glitter_rgb_curves, curve_data.SATIN_RGB_CURVE)
    # LDraw allows FRACTION as the surface fallback when volumetric grains are unavailable.
    texture_coordinate = nw.new_node(Nodes.TextureCoord, label="LDraw Glitter Coordinates")
    custom_glitter = nw.new_node(
        nodegroup_ldraw_particle_distribution(),
        input_kwargs={
            "Position": texture_coordinate.outputs["Object"],
            "Minimum Size": group_input.outputs["Minimum Particle Size"],
            "Maximum Size": group_input.outputs["Maximum Particle Size"],
            "Fraction": group_input.outputs["Chip Fraction"],
        },
        label="LDraw Glitter Surface Distribution",
    )
    glitter_weight = nw.new_node(
        Nodes.Mix,
        input_kwargs={
            "Factor": group_input.outputs["LDraw Distribution"],
            "A": glitter_rgb_curves,
            "B": custom_glitter.outputs["Mask"],
        },
        label="Glitter Weight",
        attrs={"data_type": "FLOAT"},
    )

    principled_bsdf_1 = fresnel_glass_mix(
        nw,
        group_input.outputs["WhiteColor"],
        0.0500,
        glitter_trans_group_normal_group,
        "GlitterTransGroupBody",
    )

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["ChipColor"],
            "Metallic": 0.414716,
            "Roughness": 0.0000,
            "IOR": 1.4900,
            "Normal": glitter_trans_group_normal_group,
            "Transmission Weight": 1.0000,
            "Alpha": group_input.outputs["Chip Alpha"],
            "Emission Color": group_input.outputs["ChipColor"],
            "Emission Strength": group_input.outputs["Chip Luminance"],
        },
        label="GlitterTransGroupPrincipledBSDF2",
        attrs=_GGX,
    )

    glitter_trans_group_mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": glitter_weight.outputs["Result"], 1: principled_bsdf_1, 2: principled_bsdf},
        label="GlitterTransGroupMixShader",
    )
    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Shader": glitter_trans_group_mix_shader,
            "Volume": glitter_trans_group_volume_absorption,
            "Normal": glitter_trans_group_normal_group,
        },
        label="GroupOutput",
    )


@node_utils.to_nodegroup("Mat_Chrome")
def nodegroup_chrome_group(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000))],
    )

    chrome_normal = _rounded_normal(nw, "ChromeGroupRoundingEdgeNormal")

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["BaseColor"],
            "Metallic": 1.0000,
            "Roughness": 0.0500,
            "IOR": 1.4500,
            "Normal": chrome_normal,
        },
        attrs=_GGX,
    )

    chrome_glossy_bsdf = nw.new_node(
        "ShaderNodeBsdfAnisotropic",
        input_kwargs={
            "Roughness": 0.0500,
            "Normal": chrome_normal,
        },
        label="ChromeGroupGlossyBSDF",
        attrs=_GGX,
    )

    chrome_group_mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": 0.1000, 1: principled_bsdf, 2: chrome_glossy_bsdf},
        label="ChromeGroupMixShader",
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"BSDF": chrome_group_mix_shader}, label="GroupOutput")


@node_utils.to_nodegroup("Mat_Chrome Antique")
def nodegroup_chrome_antique_group(nw: NodeWrangler):
    coord_fix = nw.new_node(Nodes.TextureCoord, label="Antique Chrome Texture Coordinates")
    chrome_antique_mix_factor = 0.728675

    # All four Principled lobes share one deterministic rounded-normal raycast instance.
    chrome_antique_normal = _rounded_normal(nw, "ChromeAntiqueGroupRoundingEdgeNormal")

    principled_bsdf_3 = new_principled(
        nw,
        input_kwargs={
            "Base Color": (0.63, 0.51, 0.21, 1.0000),
            "Metallic": 0.9000,
            "Roughness": 0.3000,
            "IOR": 1.4500,
            "Normal": chrome_antique_normal,
        },
        attrs=_GGX,
    )

    chrome_antique_group_geometry = nw.new_node(Nodes.NewGeometry, label="ChromeAntiqueGroupGeometry")

    chrome_antique_color_ramp = node_utils.build_color_ramp(
        nw,
        chrome_antique_group_geometry.outputs["Pointiness"],
        [position for position, _ in curve_data.CHROME_ANTIQUE_COLOR_RAMP3],
        [color for _, color in curve_data.CHROME_ANTIQUE_COLOR_RAMP3],
        mode="RGB",
    )
    chrome_antique_color_ramp.name = "ChromeAntiqueGroupColorRamp3"
    chrome_antique_color_ramp.label = "ChromeAntiqueGroupColorRamp3"

    chrome_antique_noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={"Vector": coord_fix.outputs["Object"], "Scale": 3.0000, "Detail": 0.0000},
        label="ChromeAntiqueGroupNoiseTexture1",
    )

    chrome_antique_rgb_curves = nw.new_node(
        Nodes.RGBCurve,
        input_kwargs={"Factor": 0.9000, "Color": chrome_antique_noise_texture.outputs["Factor"]},
        label="ChromeAntiqueGroupRGBCurves",
    )
    assign_rgb_curves(chrome_antique_rgb_curves, curve_data.CHROME_ANTIQUE_RGB_CURVE)

    group_input = nw.new_node(
        Nodes.GroupInput,
        label="GroupInput",
        expose_input=[("NodeSocketColor", "BaseColor", (0.0000, 0.0000, 0.0000, 1.0000))],
    )

    principled_bsdf = new_principled(
        nw,
        input_kwargs={
            "Base Color": group_input.outputs["BaseColor"],
            "Metallic": 1.0000,
            "Roughness": 0.1000,
            "IOR": 1.4500,
            "Normal": chrome_antique_normal,
        },
        attrs=_GGX,
    )

    principled_bsdf_2 = new_principled(
        nw,
        input_kwargs={
            "Base Color": (0.44882873, 0.33631286, 0.14206293, 1.0000),
            "Metallic": 1.0000,
            "Roughness": 0.1000,
            "IOR": 1.4500,
            "Normal": chrome_antique_normal,
        },
        attrs=_GGX,
    )

    chrome_antique_mix_shader3 = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": chrome_antique_rgb_curves, 1: principled_bsdf, 2: principled_bsdf_2},
        label="ChromeAntiqueGroupMixShader3",
    )

    principled_bsdf_1 = new_principled(
        nw,
        input_kwargs={
            "Base Color": (0.0000, 0.0000, 0.0000, 1.0000),
            "Metallic": 0.8000,
            "IOR": 1.4500,
            "Normal": chrome_antique_normal,
        },
        attrs=_GGX,
    )

    chrome_antique_mix_shader1 = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Factor": chrome_antique_color_ramp.outputs["Color"],
            1: chrome_antique_mix_shader3,
            2: principled_bsdf_1,
        },
        label="ChromeAntiqueGroupMixShader1",
    )

    chrome_antique_mix_shader2 = nw.new_node(
        Nodes.MixShader,
        input_kwargs={"Factor": chrome_antique_mix_factor, 1: principled_bsdf_3, 2: chrome_antique_mix_shader1},
        label="ChromeAntiqueGroupMixShader2",
    )

    nw.new_node(Nodes.GroupOutput, input_kwargs={"BSDF": chrome_antique_mix_shader2}, label="GroupOutput")
