"""

Utilties for working with my own scene JSON format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

# Model fields:

# * name - name of model
# * filename - filename to load, usually obj
# * copy - name of other model to copy mesh data from
# * color - default / overall color for the model
# * material - material information
# * auto_smooth_angle - if present, enable auto smoothing at the specified angle
# * transformation - object with 'rotation', 'translation', and 'scale' fields.
#     Defaults to identity transformation if not present.
# * hide - if true, hide the model in both viewport and render.
# * props - optional properties such as modifiers
#     * blender:wireframe - wireframe modifier
# * children - a list of additional model objects that should be positioned
#     relative to this one

# Animation keyframe fields:

# * transformation
# * hide

# Material fields:
# * name
# * copy
# * color
# * diffuse
# * emission
# * python
#    * func
#    * paths



# TODO: make all these field names constants, for goodness sakes


from typing import Any, Dict, Optional
import math

import bpy

from modeling import blender, materials


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
        bpy.data.materials.remove(obj.data.materials[0])
        obj.name = name
    else:
        instance_name = model_config.get('instance')
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

        mat = model_config.get('material')

        if mat is not None:

            instance_name = mat.get('instance')
            mat_python = mat.get('python')

            if instance_name is not None:
                # instance another material

                mat_copy = bpy.data.materials.get(instance_name)
                if mat_copy is not None:
                    material = mat_copy
                    # bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)
                else:
                    print(f'material {instance_name} not found to instance')

            elif mat_python is not None:
                # load a material from a python function

                blender.select(obj)

                material = materials.material_python(
                    name=mat['name'],
                    obj=obj,
                    func_desc=mat_python['func'],
                    paths=mat_python['paths']
                )

            else:
                # create a new material for the object

                material = bpy.data.materials.new(name=name)
                material.use_nodes = True

                bsdf = material.node_tree.nodes['Principled BSDF']

                color = model_config.get('color')
                if color is not None:
                    bsdf.inputs['Base Color'].default_value = color

                diffuse = mat.get('diffuse')
                if diffuse is not None:
                    bsdf.inputs['Base Color'].default_value = diffuse
                emission = mat.get('emission')
                if emission is not None:
                    bsdf.inputs['Emission'].default_value = emission

            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)

        # additional properties
        props = model_config.get('props')

        if props is not None:

            # extreme hack
            materials_library = props.get('blender:materials_library')
            if materials_library is not None:

                # TODO: make this more robust
                bpy.context.scene.matlib.lib_index = materials_library['lib_index']
                bpy.context.scene.matlib.mat_index = materials_library['mat_index']
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

            disable_shadows = props.get('blender:shadow_disable', False)
            if disable_shadows:
                obj.cycles_visibility.shadow = False
                obj.active_material.shadow_method = 'NONE'

            emission_strength = props.get('blender:emission_strength')
            if emission_strength is not None:
                bsdf.inputs['Emission Strength'].default_value = emission_strength

            alpha = props.get('blender:alpha')
            if alpha is not None:
                bsdf.inputs['Alpha'].default_value = alpha

            subsurf = props.get('blender:subsurface')
            if subsurf is not None:
                blender.add_subsurface(
                    obj,
                    levels=subsurf['levels'],
                    render_levels=subsurf['render_levels'],
                    use_adaptive_subdivision=subsurf['use_adaptive_subdivision']
                )

    if model_config.get('hide', False):
        # obj.hide_set(True)
        obj.hide_viewport = True
        obj.hide_render = True

    children = model_config.get('children', [])
    for child in children:
        add_model(child, obj)

    return obj


def set_transformation(
        obj: bpy.types.Object,
        transf: Dict[str, str]) -> None:
    """set the transformation of an object"""

    scale = transf.get('scale')
    if scale is not None:
        obj.scale = scale

    trans = transf.get('translation')
    if trans is not None:
        obj.location = trans

    rot = transf.get('rotation')
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

        translation = transformation.get('translation')
        if translation is not None:
            obj.location = tuple(translation)
            obj.keyframe_insert('location')

        rotation = transformation.get('rotation')
        if rotation is not None:
            obj.rotation_quaternion = tuple(rotation)
            obj.keyframe_insert('rotation_quaternion')

        scale = transformation.get('scale')
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
