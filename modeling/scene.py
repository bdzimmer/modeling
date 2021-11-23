"""

Utilties for working with my own scene JSON format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.


import math
from typing import Any, Dict, Optional

import bpy
from bpy import types as btypes

from modeling import blender, materials, profiler


PROFILER = profiler.Profiler()


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

    VALUE = 'value'

    BLENDER_TRIS_TO_QUADS = 'blender:tris_to_quads'

    BLENDER_WIREFRAME = 'blender:wireframe'
    BLENDER_WIREFRAME_THICKNESS = 'thickness'
    BLENDER_WIREFRAME_USE_EVEN_OFFSET = 'use_even_offset'

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
    INSTANCE = 'instance'
    PYTHON = 'python'
    MATLIB = 'matlib'

    PYTHON_FUNC = 'func'
    PYTHON_PATHS = 'paths'

    MATLIB_LIB_INDEX = 'lib_index'
    MATLIB_MAT_INDEX = 'mat_index'

    NAME = 'name'
    COPY = 'copy'
    COLOR = 'color'
    DIFFUSE = 'diffuse'
    EMISSION = 'emission'
    UPDATES = 'updates'


# TODO: figure out how to control the name of the default collection
DEFAULT_COLLECTION = 'Collection'


def add_model(
        model_config: Dict[str, Any],
        parent: Optional[bpy.types.Object]) -> bpy.types.Object:
    """add a model to the blender scene"""

    # Note that some logic in here for material assumes that model names are unique.

    name = model_config[ModelKeys.NAME]

    PROFILER.tick('add - load / instance / copy')

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

    PROFILER.tock('add - load / instance / copy')

    PROFILER.tick('add - other')

    if parent is not None:
        PROFILER.tick('add - other - parent')
        obj.parent = parent
        PROFILER.tock('add - other - parent')

    transformation = model_config.get(ModelKeys.TRANSFORMATION)
    if transformation is not None:
        PROFILER.tick('add - other - transformation')
        set_transformation(obj, transformation)
        PROFILER.tock('add - other - transformation')

    if obj.data is not None:

        # enable smooth shading
        auto_smooth_angle = model_config.get(ModelKeys.AUTO_SMOOTH_ANGLE)
        if auto_smooth_angle is not None:
            PROFILER.tick('add - other - smooth')
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = auto_smooth_angle * math.pi / 180.0
            bpy.ops.object.shade_smooth()
            PROFILER.tock('add - other - smooth')

        mat = model_config.get(ModelKeys.MATERIAL)

        if mat is not None:

            PROFILER.tick('add - other - material')

            material = add_material(mat, obj, model_config)
            if material is not None:
                if obj.data.materials:
                    obj.data.materials[0] = material
                else:
                    obj.data.materials.append(material)

            PROFILER.tock('add - other - material')

        # additional properties

        props = model_config.get(ModelKeys.PROPS)

        if props is not None:
            PROFILER.tick('add - other - properties')
            for prop in props:
                set_prop(obj, prop)
            PROFILER.tock('add - other - properties')

    if model_config.get('hide', False):
        # obj.hide_set(True)
        PROFILER.tick('add - other - hide')
        obj.hide_viewport = True
        obj.hide_render = True
        PROFILER.tock('add - other - hide')

    # tock before recurse
    PROFILER.tock('add - other')

    children = model_config.get('children', [])
    for child in children:
        add_model(child, obj)

    return obj


def add_material(
        mat: Dict,
        obj: btypes.Object,
        model_config: Optional[Dict]) -> Optional[btypes.Material]:

    instance_dict = mat.get(MaterialKeys.INSTANCE)
    mat_python = mat.get(MaterialKeys.PYTHON)
    matlib_dict = mat.get(MaterialKeys.MATLIB)

    if instance_dict is not None:
        # instance a material that already exists in the scene

        instance_name = instance_dict['name']
        material = bpy.data.materials.get(instance_name)
        if material is None:
            print(f'material {instance_name} not found to instance')
            return None

        if instance_dict.get('copy', False):
            print('copying material')
            # not sure, but it appears that copy is required here
            # I think make local is required too
            material = material.copy()
            material = material.make_local()
            material.name = mat[MaterialKeys.NAME]

    elif matlib_dict is not None:

        # TODO: make this more robust
        # TODO: is a sub dictionary like this what makes the most sense?

        bpy.context.scene.matlib.lib_index = matlib_dict[MaterialKeys.MATLIB_LIB_INDEX]
        bpy.context.scene.matlib.mat_index = matlib_dict[MaterialKeys.MATLIB_MAT_INDEX]

        obj.select_set(True)
        bpy.context.scene.matlib.apply(bpy.context)

        material = obj.active_material

        # optionally duplicate instead of just instance
        # this defaults to true, since usually this is what we want
        # for materials library materials
        if matlib_dict.get('copy', True):
            mat_name = obj.name + ' - ' + material.name  # derive unique name
            material = material.make_local()  # .copy() will share animation keyframes
            material.name = mat_name
            obj.active_material = material  # this might not be necessary

        obj.select_set(False)

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

    # apply additional updates
    updates = mat.get(MaterialKeys.UPDATES)
    if updates is not None:
        for update in updates:

            value = update['value']
            node_name = update['node']
            node = material.node_tree.nodes.get(node_name)
            if node is None:
                for node_cur in material.node_tree.nodes:
                    if node_cur.label == node_name:
                        node = node_cur
                        break

            node_input_name = update.get('input')
            node_output_name = update.get('output')
            if node_input_name is not None:
                node_input = node.inputs[node_input_name]
                node_input.default_value = value
            elif node_output_name is not None:
                node_output = node.outputs[node_output_name]
                node_output.default_value = value

    return material


def set_prop(
        obj: bpy.types.Object,
        prop_dict: Dict[str, Any]) -> None:

    """
    Set a property on an object. This can be various things,
    modifiers, settings changes, etc."""

    prop_type = prop_dict['type']

    # if prop_type == PropsKeys.BLENDER_MATERIALS_LIBRARY:
    #     # TODO: make this more robust
    #     bpy.context.scene.matlib.lib_index = prop_dict[PropsKeys.BLENDER_MATERIALS_LIBRARY_LIB_INDEX]
    #     bpy.context.scene.matlib.mat_index = prop_dict[PropsKeys.BLENDER_MATERIALS_LIBRARY_MAT_INDEX]
    #     obj.select_set(True)
    #     bpy.context.scene.matlib.apply(bpy.context)
    #
    #     if prop_dict.get('copy', True):
    #         mat = obj.active_material
    #         mat_name = obj.name + ' - ' + mat.name
    #         # mat = mat.copy()
    #         mat = mat.make_local()  # .copy() will share animation keyframes
    #         mat.name = mat_name
    #
    #         obj.active_material = mat
    #
    #     obj.select_set(False)

    if prop_type == PropsKeys.BLENDER_TRIS_TO_QUADS:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.editmode_toggle()

    elif prop_type == PropsKeys.BLENDER_WIREFRAME:

        # required to add modifiers
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='WIREFRAME')
        obj.modifiers['Wireframe'].thickness = prop_dict.get(PropsKeys.BLENDER_WIREFRAME_THICKNESS, 0.02)
        obj.modifiers['Wireframe'].use_even_offset = prop_dict.get(PropsKeys.BLENDER_WIREFRAME_USE_EVEN_OFFSET, True)

        # TODO: remove this functionality from this property
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.editmode_toggle()

    elif prop_type == PropsKeys.BLENDER_SHADOW_DISABLE:
        obj.cycles_visibility.shadow = False
        obj.active_material.shadow_method = 'NONE'

    elif prop_type == PropsKeys.BLENDER_EMISSION_STRENGTH:
        mat = obj.active_material
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Emission Strength'].default_value = prop_dict[PropsKeys.VALUE]

    elif prop_type == PropsKeys.BLENDER_ALPHA:
        mat = obj.active_material
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Alpha'].default_value = prop_dict[PropsKeys.VALUE]

    elif prop_type == PropsKeys.BLENDER_SUBSURFACE:
        blender.add_subsurface(
            obj,
            levels=prop_dict[PropsKeys.BLENDER_SUBSURFACE_LEVELS],
            render_levels=prop_dict[PropsKeys.BLENDER_SUBSURFACE_RENDER_LEVELS],
            use_adaptive_subdivision=prop_dict[PropsKeys.BLENDER_SUBSURFACE_USE_ADAPTIVE_SUBDIVISION]
        )

    elif prop_type == PropsKeys.BLENDER_BEVEL:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='BEVEL')

        # TODO: write this in a better way
        obj.modifiers['Bevel'].width = prop_dict[PropsKeys.BLENDER_BEVEL_WIDTH]
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

    PROFILER.tick('obj')

    bpy.ops.import_scene.obj(
        filepath=filename,
        use_smooth_groups=False,  # for now
        split_mode='OFF',
        axis_forward='Y',
        axis_up='Z')
    obj = bpy.context.selected_objects[0]

    PROFILER.tock('obj')

    return obj
