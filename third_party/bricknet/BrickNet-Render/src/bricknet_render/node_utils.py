"""Construct Blender shader-node graphs."""

# Infinigen-derived code below: BSD-3-Clause, Copyright (C) 2023-2024 Princeton University.

import logging
from functools import wraps

import bpy

logger = logging.getLogger(__name__)
NODE_GROUP_OWNER_PROP = "_bricknet_render_node_group"


class Nodes:
    """
    An enum for all node types.
    Pass these as the first argument to nw.new_node(...)

    Organized according to the Blender UI's Shift-A 'Add Node' Menu .
    """

    Mix = "ShaderNodeMix"

    # Color Menu
    ColorRamp = "ShaderNodeValToRGB"
    RGBCurve = "ShaderNodeRGBCurve"
    SeparateColor = "ShaderNodeSeparateColor"

    # Input
    GroupInput = "NodeGroupInput"
    RGB = "ShaderNodeRGB"
    Value = "ShaderNodeValue"
    TextureCoord = "ShaderNodeTexCoord"
    LightPath = "ShaderNodeLightPath"

    # Output Menu
    GroupOutput = "NodeGroupOutput"
    MaterialOutput = "ShaderNodeOutputMaterial"

    # Vector
    SeparateXYZ = "ShaderNodeSeparateXYZ"
    CombineXYZ = "ShaderNodeCombineXYZ"

    # Math
    VectorMath = "ShaderNodeVectorMath"
    Math = "ShaderNodeMath"

    # Texture
    NoiseTexture = "ShaderNodeTexNoise"
    VoronoiTexture = "ShaderNodeTexVoronoi"
    WhiteNoiseTexture = "ShaderNodeTexWhiteNoise"

    # Shaders
    MixShader = "ShaderNodeMixShader"
    DiffuseBSDF = "ShaderNodeBsdfDiffuse"
    PrincipledBSDF = "ShaderNodeBsdfPrincipled"
    TranslucentBSDF = "ShaderNodeBsdfTranslucent"
    Emission = "ShaderNodeEmission"
    Fresnel = "ShaderNodeFresnel"
    NewGeometry = "ShaderNodeNewGeometry"
    RefractionBSDF = "ShaderNodeBsdfRefraction"
    GlossyBSDF = "ShaderNodeBsdfAnisotropic"
    LayerWeight = "ShaderNodeLayerWeight"


# Certain nodes should only be created once. This list defines which ones.
SINGLETON_NODES = [
    Nodes.GroupInput,
    Nodes.GroupOutput,
    Nodes.MaterialOutput,
]

NODECLASSES = [k for k in dir(bpy.types) if "NodeSocket" in k]


def infer_output_socket(item):
    """
    Figure out if `item` somehow represents a node with an output we can use.
    If so, return that output socket
    """

    if isinstance(item, bpy.types.NodeSocket):
        res = item
    elif isnode(item):
        # take the first active socket
        try:
            res = next(o for o in item.outputs if o.enabled)
        except StopIteration:
            raise ValueError(f"Attempted to get output socket for {item} but none are enabled!")
    else:
        return None

    if not res.enabled:
        raise ValueError(
            f"Attempted to use output socket {res} of node {res.node} which is not enabled. "
            "Please check your attrs are correct, or be more specific about the socket to use"
        )

    return res


def infer_input_socket(node, input_socket_name):
    if isinstance(input_socket_name, str):
        try:
            input_socket = next(i for i in node.inputs if i.name == input_socket_name and i.enabled)
        except StopIteration:
            input_socket = node.inputs[input_socket_name]
    else:
        input_socket = node.inputs[input_socket_name]

    if not input_socket.enabled:
        logger.warning(
            f"Attempted to use ({input_socket.name=},{input_socket.type=}) of {node.name=}, but it was "
            f"disabled. Either change attrs={{...}}, "
            f"change the socket index, or specify the socket by name (assuming two enabled sockets don't "
            f"share a name)."
            f"The input sockets are "
            f"{[(i.name, i.type, ('ENABLED' if i.enabled else 'DISABLED')) for i in node.inputs]}.",
        )

    return input_socket


def isnode(x):
    return isinstance(x, bpy.types.NodeInternal)


def ng_inputs(node_group):
    return {s.name: s for s in node_group.interface.items_tree if s.in_out == "INPUT"}


class NodeWrangler:
    def __init__(self, node_group):
        if not issubclass(type(node_group), bpy.types.NodeTree):
            raise ValueError(f"Couldn't initialize NodeWrangler with {node_group=}, {type(node_group)=}")
        self.node_group = node_group

        self.nodes = self.node_group.nodes
        self.links = self.node_group.links

    def new_node(
        self,
        node_type,
        attrs=None,
        input_kwargs=None,
        label=None,
        expose_input=None,
    ):
        if input_kwargs is None:
            input_kwargs = {}

        if attrs is None:
            attrs = {}

        node = self._make_node(node_type)

        if label is not None:
            node.label = label
            node.name = label

        for key_path, val in attrs.items():
            keys = key_path.split(".")
            obj = node
            for key in keys[:-1]:
                obj = getattr(obj, key)
            setattr(obj, keys[-1], val)

        for input_socket_name, input_item in input_kwargs.items():
            if input_item is None:
                continue
            if node_type == Nodes.GroupOutput:
                assert not isinstance(input_socket_name, int), (
                    f"Attribute inputs to group output nodes must be given a string "
                    f"name, integer name "
                    f"{input_socket_name} will not suffice"
                )
                assert not isinstance(input_item, list), "Multi-input sockets to GroupOutput nodes are impossible"
                if input_socket_name not in node.inputs:
                    nodeclass = infer_output_socket(input_item).bl_idname
                    if nodeclass == "NodeSocketVectorEuler":
                        nodeclass = "NodeSocketVector"
                    self.node_group.interface.new_socket(name=input_socket_name, in_out="OUTPUT", socket_type=nodeclass)
                    assert input_socket_name in node.inputs and node.inputs[input_socket_name].enabled

            input_socket = infer_input_socket(node, input_socket_name)
            self.connect_input(input_socket, input_item)

        if expose_input is not None:
            names = [v[1] for v in expose_input]
            if len(set(names)) != len(names):
                raise ValueError(f"expose_input with {names} features duplicate entries.")
            for inp in expose_input:
                nodeclass, name, val = inp
                self.expose_input(name, val=val, dtype=nodeclass)

        return node

    def expose_input(self, name, val=None, dtype=None):
        """
        Expose an input to the nodegroups interface, making it able to be specified externally
        """

        group_input = self.new_node(Nodes.GroupInput)  # will reuse singleton

        if name not in ng_inputs(self.node_group):
            if dtype not in NODECLASSES:
                raise ValueError(f"Could not parse {dtype=}")
            inp = self.node_group.interface.new_socket(name, in_out="INPUT", socket_type=dtype)

            if val is not None:
                if not hasattr(inp, "default_value") or inp.default_value is None:
                    raise ValueError(f"expose_input() received {val=} but inp {inp} does not expect a default_value")
                # Cast val only when necessary, and only when type(val) won't crash.
                to_type = type(inp.default_value)
                if to_type not in (bpy.types.bpy_prop_array, bpy.types.bpy_prop):
                    val = to_type(val)
                inp.default_value = val

        assert len([o for o in group_input.outputs if o.name == name]) == 1
        return group_input.outputs[name]

    def _update_socket(self, input_socket, input_item):
        output_socket = infer_output_socket(input_item)

        if output_socket is None and hasattr(input_socket, "default_value"):
            # No node output was found, so assign the value directly to the input socket.
            try:
                input_socket.default_value = input_item
                return
            except TypeError as e:
                print(f"TypeError while assigning {input_item=} as default_value for {input_socket.name}")
                raise e

        self.links.new(output_socket, input_socket)

    def connect_input(self, input_socket, input_item):
        if isinstance(input_item, list) and any(infer_output_socket(i) is not None for i in input_item):
            if not input_socket.is_multi_input:
                raise ValueError(
                    f"list of sockets {input_item} is not valid to connect to {input_socket} as it is not a "
                    f"valid multi-input socket"
                )
            for inp in input_item:
                self._update_socket(input_socket, inp)
        else:
            self._update_socket(input_socket, input_item)

    def _make_node(self, node_type):
        if node_type in SINGLETON_NODES:
            # for nodes in the singletons list, we should reuse an existing one if it exists
            try:
                node = next(n for n in self.nodes if n.bl_idname == node_type)
            except StopIteration:
                node = self.nodes.new(node_type)
        elif isinstance(node_type, bpy.types.ShaderNodeTree):
            node = self.nodes.new("ShaderNodeGroup")
            node.node_tree = node_type
        else:
            node = self.nodes.new(node_type)

        return node


def to_nodegroup(name):
    """Cache one package-owned shader node group by Blender datablock name."""

    def registration_fn(fn):
        @wraps(fn)
        def init_fn(*args, **kwargs):
            existing = bpy.data.node_groups.get(name)
            if existing is not None:
                if existing.get(NODE_GROUP_OWNER_PROP) != name:
                    raise RuntimeError("node-group name %s is already in use" % name)
                return existing
            ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
            ng[NODE_GROUP_OWNER_PROP] = name
            try:
                fn(NodeWrangler(ng), *args, **kwargs)
            except Exception:
                bpy.data.node_groups.remove(ng)
                raise
            return ng

        return init_fn

    return registration_fn


def assign_curve(c, points):
    for i, p in enumerate(points):
        if i < 2:
            c.points[i].location = p
        else:
            c.points.new(*p)


def build_color_ramp(nw, x, positions, colors, mode="HSV"):
    cr = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": x})
    cr.color_ramp.color_mode = mode
    elements = cr.color_ramp.elements
    size = len(positions)
    assert len(colors) == size
    if size > 2:
        for _ in range(size - 2):
            elements.new(0)
    for i, (p, c) in enumerate(zip(positions, colors)):
        elements[i].position = p
        elements[i].color = c
    return cr


def shaderfunc_to_material(shader_func, *args, name=None, **kwargs):
    """Convert a shader_func(nw) directly to a bpy.data.material."""

    if name is None:
        name = shader_func.__name__

    material = bpy.data.materials.new(name=name)
    material.node_tree.nodes.remove(material.node_tree.nodes["Principled BSDF"])  # remove the default BSDF

    nw = NodeWrangler(material.node_tree)

    shader_func(nw, *args, **kwargs)

    return material
