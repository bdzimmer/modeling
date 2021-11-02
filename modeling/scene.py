"""

Utilties for working with my own scene JSON format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.


import math
from typing import Any, Dict, Optional

import bpy
from bpy import types as btypes

from modeling import blender, materials


class ModelKeys:
    """keys for model dict"""

    NAME = 'name'            # name of model
    FILENAME = 'filename'    # filename to load, usually obj
    COPY = 'copy'            # name of other model to copy mesh data from
    COLOR = 'color'          # default / overall color for the model
    MATERIAL = 'material'    # material information

    # if present, enable auto smoothing at the specified angle
    AUTO_SMOOTH_ANGLE = 'auto_smooth_angle'

    # dict with 'rotation', 'translation', and 'scale' fields.
    TRANSFORMATION = 'transformation'

    HIDE = 'hide'                                  # if true, hide the model
    PROPS = 'props'                                # optional properties, such as modifiers

    INSTANCE = 'instance'

    # a list of additional model objects that should be positioned
    # relative to this one
    CHILDREN = 'children'


class PropsKeys:
    """keys for properties"""

    BLENDER_TRIS_TO_QUADS = 'blender:tris_to_quads'

    BLENDER_WIREFRAME = 'blender:wireframe'
    BLENDER_WIREFRAME_THICKNESS = 'thickness'
    BLENDER_WIREFRAME_USE_EVEN_OFFSET = 'use_even_offset'

    BLENDER_MATERIALS_LIBRARY = 'blender:materials_library'
    BLENDER_MATERIALS_LIBRARY_LIB_INDEX = 'lib_index'
    BLENDER_MATERIALS_LIBRARY_MAT_INDEX = 'mat_index'

    BLENDER_SHADOW_DISABLE = 'blender:shadow_disable'

    BLENDER_EMISSION_STRENGTH = 'blender:emission_strength'

    BLENDER_ALPHA = 'blender:alpha'

    BLENDER_SUBSURFACE = 'blender:subsurface'
    BLENDER_SUBSURFACE_LEVELS = 'levels'
    BLENDER_SUBSURFACE_RENDER_LEVELS = 'render_levels'
    BLENDER_SUBSURFACE_USE_ADAPTIVE_SUBDIVISION = 'use_adaptive_subdivision'
    BLENDER_BEVEL = 'blender:bevel'
    BLENDER_BEVEL_WIDTH = 'width'


class TransformationKeys:
    """keys for transformations"""
    ROTATION = 'rotation'
    TRANSLATION = 'translation'
    SCALE = 'scale'


class KeyframeKeys:
    """keys for animation keyframes"""
    TRANSFORMATION = 'transformation'
    HIDE = 'hide'


class MaterialKeys:
    """keys for materials"""

    NAME = 'name'
    COPY = 'copy'  # TODO: rename to something like 'deepcopy' or 'instance'
    INSTANCE = 'instance'
    COLOR = 'color'
    DIFFUSE = 'diffuse'
    EMISSION = 'emission'
    PYTHON = 'python'
    PYTHON_FUNC = 'func'
    PYTHON_PATHS = 'paths'


# TODO: figure out how to control the name of the default collection
DEFAULT_COLLECTION = 'Collection'


def add_model(
        model_config: Dict[str, Any],
        parent: Optional[bpy.types.Object]) -> bpy.types.Object:
    """add a model to the blender scene"""

    # Note that some logic in here for material assumes that model names are unique.

    name = model_config[ModelKeys.NAME]

    model_filename = model_config.get(ModelKeys.FILENAME)
    if model_filename is not None:
        obj = import_obj(model_config[ModelKeys.FILENAME])
        bpy.data.materials.remove(obj.data.materials[0])
        obj.name = name
    else:
        instance_name = model_config.get(ModelKeys.INSTANCE)
        if instance_name is None:
            # create empty
            obj = bpy.data.objects.new(name, None)
            bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)
        else:
            # instance another object
            obj_copy = bpy.data.objects.get(instance_name)
            if obj_copy is not None:
                obj = bpy.data.objects.new(name, obj_copy.data)
                bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)
            else:
                print(f'object {instance_name} not found to instance')

    if parent is not None:
        obj.parent = parent

    transformation = model_config.get(ModelKeys.TRANSFORMATION)
    if transformation is not None:
        set_transformation(obj, transformation)

    # enable smooth shading
    auto_smooth_angle = model_config.get(ModelKeys.AUTO_SMOOTH_ANGLE)
    if auto_smooth_angle is not None:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = auto_smooth_angle * math.pi / 180.0
        bpy.ops.object.shade_smooth()

    if obj.data is not None:

        mat = model_config.get(ModelKeys.MATERIAL)

        if mat is not None:
            material = add_material(mat, obj, model_config)
            if material is not None:
                if obj.data.materials:
                    obj.data.materials[0] = material
                else:
                    obj.data.materials.append(material)

        # additional properties
        props = model_config.get(ModelKeys.PROPS)

        if props is not None:
            set_props(obj, props)

    if model_config.get('hide', False):
        # obj.hide_set(True)
        obj.hide_viewport = True
        obj.hide_render = True

    children = model_config.get('children', [])
    for child in children:
        add_model(child, obj)

    return obj


def add_material(
        mat: Dict,
        obj: btypes.Object,
        model_config: Optional[Dict]) -> Optional[btypes.Material]:

    instance_name = mat.get(MaterialKeys.INSTANCE)
    mat_python = mat.get(MaterialKeys.PYTHON)

    if instance_name is not None:
        # instance another material

        mat_copy = bpy.data.materials.get(instance_name)
        if mat_copy is not None:
            material = mat_copy
            # bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)
        else:
            print(f'material {instance_name} not found to instance')
            return None

    elif mat_python is not None:
        # load a material from a python function

        blender.select(obj)

        material = materials.material_python(
            name=mat[MaterialKeys.NAME],
            obj=obj,
            func_desc=mat_python[MaterialKeys.PYTHON_FUNC],
            paths=mat_python[MaterialKeys.PYTHON_PATHS]
        )

    else:
        # create a new material for the object

        material = bpy.data.materials.new(name=obj.name)
        material.use_nodes = True

        bsdf = material.node_tree.nodes['Principled BSDF']

        color = model_config.get(ModelKeys.COLOR)
        if color is not None:
            bsdf.inputs['Base Color'].default_value = color

        diffuse = mat.get('diffuse')
        if diffuse is not None:
            bsdf.inputs['Base Color'].default_value = diffuse
        emission = mat.get('emission')
        if emission is not None:
            bsdf.inputs['Emission'].default_value = emission

    return material


def set_props(
        obj: bpy.types.Object,
        props: Dict[str, Any]) -> None:

    # materials library
    materials_library = props.get(PropsKeys.BLENDER_MATERIALS_LIBRARY)
    if materials_library is not None:

        # TODO: make this more robust
        bpy.context.scene.matlib.lib_index = materials_library[PropsKeys.BLENDER_MATERIALS_LIBRARY_LIB_INDEX]
        bpy.context.scene.matlib.mat_index = materials_library[PropsKeys.BLENDER_MATERIALS_LIBRARY_MAT_INDEX]
        obj.select_set(True)
        bpy.context.scene.matlib.apply(bpy.context)

        if materials_library.get('copy', True):
            mat = obj.active_material
            mat_name = obj.name + ' - ' + mat.name
            # mat = mat.copy()
            mat = mat.make_local()  # .copy() will share animation keyframes
            mat.name = mat_name

            obj.active_material = mat

        obj.select_set(False)

    if props.get(PropsKeys.BLENDER_TRIS_TO_QUADS, False):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.editmode_toggle()

    wireframe = props.get(PropsKeys.BLENDER_WIREFRAME)
    if wireframe is not None:
        # required to add modifiers
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='WIREFRAME')
        obj.modifiers['Wireframe'].thickness = wireframe.get(PropsKeys.BLENDER_WIREFRAME_THICKNESS, 0.02)
        obj.modifiers['Wireframe'].use_even_offset = wireframe.get(PropsKeys.BLENDER_WIREFRAME_USE_EVEN_OFFSET, True)

        # TODO: remove this functionality from this property
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.editmode_toggle()

    disable_shadows = props.get(PropsKeys.BLENDER_SHADOW_DISABLE, False)
    if disable_shadows:
        obj.cycles_visibility.shadow = False
        obj.active_material.shadow_method = 'NONE'

    emission_strength = props.get(PropsKeys.BLENDER_EMISSION_STRENGTH)
    if emission_strength is not None:
        mat = obj.active_material
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Emission Strength'].default_value = emission_strength

    alpha = props.get(PropsKeys.BLENDER_ALPHA)
    if alpha is not None:
        mat = obj.active_material
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Alpha'].default_value = alpha

    subsurf = props.get(PropsKeys.BLENDER_SUBSURFACE)
    if subsurf is not None:
        blender.add_subsurface(
            obj,
            levels=subsurf[PropsKeys.BLENDER_SUBSURFACE_LEVELS],
            render_levels=subsurf[PropsKeys.BLENDER_SUBSURFACE_RENDER_LEVELS],
            use_adaptive_subdivision=subsurf[PropsKeys.BLENDER_SUBSURFACE_USE_ADAPTIVE_SUBDIVISION]
        )

    bevel = props.get(PropsKeys.BLENDER_BEVEL)
    if bevel is not None:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='BEVEL')

        # TODO: write this in a better way
        obj.modifiers['Bevel'].width = bevel.get(PropsKeys.BLENDER_BEVEL_WIDTH)
        # obj.modifiers['Wireframe'].use_even_offset = wireframe.get(PropsKeys.BLENDER_WIREFRAME_USE_EVEN_OFFSET, True)




def set_transformation(
        obj: bpy.types.Object,
        transf: Dict[str, str]) -> None:
    """set the transformation of an object"""

    scale = transf.get(TransformationKeys.SCALE)
    if scale is not None:
        obj.scale = scale

    trans = transf.get(TransformationKeys.TRANSLATION)
    if trans is not None:
        obj.location = trans

    rot = transf.get(TransformationKeys.ROTATION)
    if rot is not None:
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


def add_keyframes(
        obj,
        transformation: Optional[Dict],
        nodes: Optional[Dict],
        hide: Optional[bool]) -> None:
    """add a keyframe based on a transformation or visibiltiy"""

    # TODO: think about combining this with modeling.scene.set_transformation
    #       like a flag that would add keyframes

    if transformation is not None:

        obj.rotation_mode = 'QUATERNION'

        translation = transformation.get(TransformationKeys.TRANSLATION)
        if translation is not None:
            obj.location = tuple(translation)
            obj.keyframe_insert('location')

        rotation = transformation.get(TransformationKeys.ROTATION)
        if rotation is not None:
            obj.rotation_quaternion = tuple(rotation)
            obj.keyframe_insert('rotation_quaternion')

        scale = transformation.get(TransformationKeys.SCALE)
        if scale is not None:
            obj.scale = tuple(scale)
            obj.keyframe_insert(data_path='scale')

    if nodes is not None:
        print('\t\t', nodes)
        material = obj.data.materials[0]
        print('\t\t', material.name)
        for (node_name, node_input_name), value in nodes.items():
            node = material.node_tree.nodes[node_name]
            node_input = node.inputs[node_input_name]
            node_input.default_value = value
            node_input.keyframe_insert(data_path='default_value')

    if hide is not None:
        obj.hide_viewport = hide
        obj.hide_render = hide
        obj.keyframe_insert(data_path='hide_viewport')
        obj.keyframe_insert(data_path='hide_render')


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
