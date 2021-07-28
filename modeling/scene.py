"""

Utilties for working with my own scene JSON format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

# Model fields:

# * name - name of model
# * filename - filename to load, usually obj
# * color - default / overall color for the model
# * auto_smooth_angle - if present, enable auto smoothing at the specified angle
# * transformation - object with 'rotation' and 'translation' fields. Defaults
#     to identity transformation if not present.
# * props - optional properties such as modifiers
#     * blender:wireframe - wireframe modifier
# * children - a list of additional model objects that should be positioned
#     relative to this one


# TODO: make all these field names constants, for goodness sakes


from typing import Any, Dict, Optional
import math

import bpy

from modeling import blender


# TODO: figure out how to control the name of the default collection
DEFAULT_COLLECTION = 'Collection'


def add_model(
        model_config: Dict[str, Any],
        parent: Optional[bpy.types.Object]) -> bpy.types.Object:
    """add a model to the blender scene"""

    # Note that some logic in here for material assumes that model names are unique.

    name = model_config['name']

    model_filename = model_config.get('filename')
    if model_filename is not None:
        obj = import_obj(model_config['filename'])
        obj.name = name
    else:
        obj = bpy.data.objects.new(name, None)
        bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)

    if parent is not None:
        obj.parent = parent

    transformation = model_config.get('transformation')
    if transformation is not None:
        set_transformation(obj, transformation)

    # enable smooth shading
    auto_smooth_angle = model_config.get('auto_smooth_angle')
    if auto_smooth_angle is not None:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = auto_smooth_angle * math.pi / 180.0
        bpy.ops.object.shade_smooth()

    if obj.data is not None:

        # create a new material for the object
        # not sure this is correct
        # material = bpy.data.materials['Default OBJ']
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True

        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        bsdf = material.node_tree.nodes['Principled BSDF']

        color = model_config.get('color')
        if color is not None:
            bsdf.inputs[0].default_value = color

        mat = model_config.get('material')
        if mat is not None:
            emission = mat.get('emission')
            if emission is not None:
                bsdf.inputs['Emission'].default_value = emission

        # additional properties
        props = model_config.get('props')

        if props is not None:

            wireframe = props.get('blender:wireframe')
            if wireframe is not None:
                # required to add modifiers
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_add(type='WIREFRAME')
                obj.modifiers['Wireframe'].thickness = wireframe.get('thickness', 0.02)
                obj.modifiers['Wireframe'].use_even_offset = wireframe.get('use_even_offset', True)
                # TODO: make this optional
                bpy.ops.object.editmode_toggle()
                bpy.ops.mesh.tris_convert_to_quads()
                bpy.ops.object.editmode_toggle()

    children = model_config.get('children', [])
    for child in children:
        add_model(child, obj)

    return obj


def set_transformation(
        obj: bpy.types.Object,
        transf: Dict[str, str]) -> None:
    """set the transformation of an object"""

    trans = transf['translation']
    rot = transf['rotation']

    obj.location = trans
    # blender.point_at(cam, root_obj, 'TRACK_NEGATIVE_Z', 'UP_Y')

    if isinstance(rot, dict):
        point_at_obj = blender.get_obj_by_name(rot['point_at'])
        blender.point_at(
            obj,
            point_at_obj,
            blender.TRACK_AXIS[rot.get('track_axis', '-z')],
            blender.UP_AXIS[rot.get('up_axis', 'y')])
    else:
        if len(rot) == 3:
            # I think Panda3D's HPR is intrinsic
            # If Blender is extrinsic, the order can just be reversed, LOL
            # rot is in HPR form
            # H -> Z
            # P -> X
            # R -> Y
            obj.rotation_mode = 'ZXY'
            obj.rotation_euler = (
                math.radians(rot[1]),  # X == (ZXY)[1]
                math.radians(rot[2]),  # Y == (ZXY)[2]
                math.radians(rot[0]),  # Z == (ZXY)[0]
            )
        else:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = rot


def import_obj(filename) -> bpy.types.Object:
    """import an obj file"""
    bpy.ops.import_scene.obj(
        filepath=filename,
        use_smooth_groups=False,  # for now
        split_mode='OFF',
        axis_forward='Y',
        axis_up='Z')
    obj = bpy.context.selected_objects[0]
    return obj
