"""

Utilties for working with my own scene JSON format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

import os
import sys

import math
from typing import Any, Dict, Optional, Union

import bpy
from bpy import types as btypes
import mathutils

from modeling import profiler
from modeling.blender import util as butil, materials
from modeling.blender.types import \
    ConfigTypes, ConfigObject, ConfigEmpty, ConfigLight, ConfigModel, ConfigInstance, \
    ConfigArmature, ConfigBone, \
    PropsKeys, TransformationKeys, MaterialKeys


PROFILER = profiler.Profiler()

# Note that these might change between blender versions, eg 'View Layer' -> 'ViewLayer'
DEFAULT_COLLECTION = 'Collection'
DEFAULT_VIEW_LAYER = 'ViewLayer'

TRIS_TO_QUADS_FACE_THRESH = 0.001


def add_model(
        model_config: ConfigObject,
        parent: Optional[bpy.types.Object]) -> Optional[bpy.types.Object]:
    """add a model to the blender scene"""

    # Note that some logic in here for material assumes that model names are unique.

    name = model_config.name

    PROFILER.tick('add - load / instance / copy')

    collection_name = model_config.collection if model_config.collection is not None else DEFAULT_COLLECTION
    print('collection:', collection_name)

    append_collection = False

    if isinstance(model_config, ConfigModel):
        # assumes .filename is defined
        if not model_config.filename.startswith('append'):
            print('importing', model_config.filename, flush=True)
            obj = import_obj(model_config.filename)
            bpy.data.materials.remove(obj.data.materials[0])
            obj.name = name
        else:
            _, filepath, directory, filename = model_config.filename.split(':')
            print('appending', filepath, directory, filename, flush=True)

            # TODO: find a way to do this without ops
            bpy.ops.wm.append(
                filepath=os.path.join(filepath, directory, filename),
                directory=os.path.join(filepath, directory),
                filename=filename
            )
            if directory == 'Object':
                obj = bpy.context.object
                obj.name = name
            elif directory == 'Collection':
                append_collection = True
                obj = butil.get_obj_by_name(name)
                # print('collection object:', obj)
                # move into appropriate collection
                if collection_name != DEFAULT_COLLECTION:
                    coll = obj.users_collection[0]  # lololol
                    # can we assume that it's currently in the default collection?
                    bpy.data.collections[DEFAULT_COLLECTION].children.unlink(coll)
                    bpy.data.collections[collection_name].children.link(coll)
            else:
                return None

    elif isinstance(model_config, ConfigInstance):
        # assumes .instance is defined
        # TODO: get rid of instance field and use something from info

        print('instancing', model_config.instance, flush=True)

        # instance another object
        obj_copy = bpy.data.objects.get(model_config.instance)
        if obj_copy is not None:
            # this instances everything
            obj = bpy.data.objects.new(name, obj_copy.data)
            bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)
        else:
            print(f'object `{model_config.instance}` not found to instance')

    elif isinstance(model_config, ConfigEmpty):
        print('creating an empty', flush=True)
        # create empty
        obj = bpy.data.objects.new(name, None)
        bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)

    elif isinstance(model_config, ConfigLight):
        print('lights not implemented yet', flush=True)
        return None

    elif isinstance(model_config, ConfigArmature):
        obj = bpy.data.objects.new(name, bpy.data.armatures.new('arm_' + name))
        obj.show_in_front = True
        # obj.data.show_names = True
        bpy.data.collections[DEFAULT_COLLECTION].objects.link(obj)

    elif isinstance(model_config, ConfigBone):
        print('bone:', model_config.name)
        print('armature parent:', parent)
        print('bone parent:', model_config.parent_bone_name)
        print('child mesh:', model_config.child_mesh_name)
        obj = None
        bpy.context.view_layer.objects.active = parent
        bpy.ops.object.editmode_toggle()
        arm: btypes.Armature = parent.data
        bone = arm.edit_bones.new(model_config.name)
        bone.roll = model_config.roll
        if model_config.connect:
            bone.use_connect = True
        bone.head = model_config.head
        bone.tail = model_config.tail
        if model_config.parent_bone_name is not None:
            parent_bone = arm.edit_bones[model_config.parent_bone_name]
            bone.parent = parent_bone
        bpy.ops.object.editmode_toggle()

        if model_config.child_mesh_name is not None:

            # select bone in armature in edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            arm.edit_bones.active = arm.edit_bones[model_config.name]

            # select child mesh and armature in object mode
            bpy.ops.object.mode_set(mode='OBJECT')
            child = butil.get_obj_by_name(model_config.child_mesh_name)

            # deselect all objects then select child and parent, making sure the
            # parent is active
            bpy.ops.object.select_all(action='DESELECT')
            child.select_set(True)
            parent.select_set(True)
            bpy.context.view_layer.objects.active = parent
            bpy.ops.object.parent_set(type='BONE', keep_transform=True)

            bpy.ops.object.select_all(action='DESELECT')

    PROFILER.tock('add - load / instance / copy')

    PROFILER.tick('add - other')

    if parent is not None and not isinstance(model_config, ConfigBone):
        PROFILER.tick('add - other - parent')
        obj.parent = parent
        PROFILER.tock('add - other - parent')

    transformation = model_config.transformation
    if transformation is not None:
        PROFILER.tick('add - other - transformation')
        set_transformation(obj, transformation)
        PROFILER.tock('add - other - transformation')

    if not isinstance(model_config, (ConfigArmature, ConfigBone)) and obj.data is not None:
        # enable smooth shading
        # auto_smooth_angle = model_config/auto_smooth_angle

        # old auto smoothing method...this should be replaced
        # auto_smooth_angle = getattr(model_config, 'auto_smooth_angle', None)
        # if auto_smooth_angle is not None:
        #     PROFILER.tick('add - other - smooth')
        #     obj.data.use_auto_smooth = True
        #     obj.data.auto_smooth_angle = auto_smooth_angle * math.pi / 180.0
        #     bpy.ops.object.shade_smooth()
        #     PROFILER.tock('add - other - smooth')

        mat = model_config.material

        if mat is not None:

            PROFILER.tick('add - other - material')

            # !!! NEW BEHAVIOR !!!
            # added this for new instancing capability
            # we probably don't want this all the time
            obj.material_slots[0].link = 'OBJECT'

            material = add_material(mat, obj, model_config)

            if material is not None:
                if obj.data.materials:
                    obj.data.materials[0] = material
                else:
                    obj.data.materials.append(material)

                # new behavior
                obj.active_material = material

            PROFILER.tock('add - other - material')

    # additional properties
    if model_config.props is not None:
        PROFILER.tick('add - other - properties')
        for prop in model_config.props:
            print('prop:', prop['type'])
            set_prop(obj, prop)
        PROFILER.tock('add - other - properties')

    hide = model_config.hide if model_config.hide is not None else False
    if hide:
        # obj.hide_set(True)
        PROFILER.tick('add - other - hide')
        obj.hide_viewport = True
        obj.hide_render = True
        PROFILER.tock('add - other - hide')

    # move into appropriate collection and link
    # TODO: probably not correct...I think this should work for instancing too
    if collection_name != DEFAULT_COLLECTION and not append_collection:
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        bpy.data.collections[collection_name].objects.link(obj)

    # tock before recurse
    PROFILER.tock('add - other')

    if model_config.children is not None:
        for child in model_config.children:
            add_model(child, obj)

    return obj


def add_material(
        mat: Dict,
        obj: btypes.Object,
        model_config: Optional[Union[ConfigModel, ConfigInstance]]) -> Optional[btypes.Material]:

    instance_dict = mat.get(MaterialKeys.INSTANCE)
    mat_python = mat.get(MaterialKeys.PYTHON)
    matlib_dict = mat.get(MaterialKeys.MATLIB)
    assetlib_dict = mat.get(MaterialKeys.ASSETLIB)

    if instance_dict is not None:
        # instance a material that already exists in the scene

        instance_name = instance_dict['name']
        material = bpy.data.materials.get(instance_name)
        if material is None:
            print(f'material `{instance_name}` not found to instance')
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

        if MaterialKeys.MATLIB_LIB_INDEX in matlib_dict:
            bpy.context.scene.matlib.lib_index = matlib_dict[MaterialKeys.MATLIB_LIB_INDEX]
            bpy.context.scene.matlib.mat_index = matlib_dict[MaterialKeys.MATLIB_MAT_INDEX]
        else:
            materials.matlib_select(
                matlib_dict[MaterialKeys.MATLIB_LIB_NAME],
                matlib_dict[MaterialKeys.MATLIB_MAT_NAME]
            )

        obj.select_set(True)
        bpy.context.scene.matlib.apply(bpy.context)

        material = obj.active_material

        # optionally duplicate instead of just instance
        # this defaults to true, since usually this is what we want
        # for materials library materials
        if matlib_dict.get(MaterialKeys.COPY, True):
            mat_name = obj.name + ' - ' + material.name  # derive unique name
            material = material.make_local()  # .copy() will share animation keyframes
            material.name = mat_name
            # obj.active_material = material  # this might not be necessary

        obj.select_set(False)

    elif mat_python is not None:
        # load a material from a python function

        butil.select(obj)

        material = materials.material_python(
            name=mat[MaterialKeys.NAME],
            obj=obj,
            func_desc=mat_python[MaterialKeys.PYTHON_FUNC],
            paths=mat_python[MaterialKeys.PYTHON_PATHS]
        )

    elif assetlib_dict is not None:
        material = materials.find_material_assetlib(
            assetlib_dict[MaterialKeys.ASSETLIB_LIB_NAME],
            assetlib_dict[MaterialKeys.ASSETLIB_MAT_NAME])
        # note that loaded node groups are still marked as assets
        material.asset_clear()

    else:
        # create a new material for the object

        material = bpy.data.materials.new(name=obj.name)
        material.use_nodes = True

        bsdf = material.node_tree.nodes['Principled BSDF']

        color = model_config.color
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

            # this is a bit of a hack
            if isinstance(value, str) and value.startswith('object:'):
                value = butil.get_obj_by_name(value[7:])

            node_name = update['node']
            node = material.node_tree.nodes.get(node_name)
            if node is None:
                for node_cur in material.node_tree.nodes:
                    if node_cur.label == node_name:
                        node = node_cur
                        break

            node_input_name = update.get('input')
            node_output_name = update.get('output')
            node_attr_name = update.get('attr')
            if node_input_name is not None:
                node_input = node.inputs[node_input_name]
                node_input.default_value = value
            elif node_output_name is not None:
                node_output = node.outputs[node_output_name]
                node_output.default_value = value
            elif node_attr_name is not None:
                setattr(node, node_attr_name, value)

    return material


def set_prop(
        obj: bpy.types.Object,
        prop_dict: Dict[str, Any]) -> None:

    """
    Set a property on an object. This can be various things,
    modifiers, settings changes, etc."""

    prop_type = prop_dict['type']

    if obj.data is None and prop_type not in [PropsKeys.BLENDER_CHILD_OF]:
        print(f'property `{prop_type}` not allowed for empty')
        return

    if prop_type == PropsKeys.AUTO_SMOOTH_ANGLE:
        # this doesn't work, we need to use select which also does select_set
        # bpy.context.view_layer.objects.active = obj
        butil.select(obj)
        auto_smooth_angle = prop_dict[PropsKeys.AUTO_SMOOTH_ANGLE_VALUE]
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = auto_smooth_angle * math.pi / 180.0
        # print('\tshade smooth')
        res = bpy.ops.object.shade_smooth()
        # print(res, obj.select_get())
        # print('\tdone')
        butil.select_none()

    elif prop_type == PropsKeys.BLENDER_TRIS_TO_QUADS:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.tris_convert_to_quads(face_threshold=TRIS_TO_QUADS_FACE_THRESH)
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
        # obj.cycles_visibility.shadow = False
        obj.visible_shadow = False
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
        butil.add_subsurface(
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

    elif prop_type == PropsKeys.BLENDER_CHILD_OF:
        obj.constraints.new('CHILD_OF')
        child_of = obj.constraints['Child Of']
        child_of.target = butil.get_obj_by_name(prop_dict.get('object'))
        child_of.use_rotation_x = False
        child_of.use_rotation_y = False
        child_of.use_rotation_z = False
        child_of.inverse_matrix = mathutils.Matrix.Identity(4)


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
            point_at_obj = butil.get_obj_by_name(rot['point_at'])
            butil.point_at(
                obj,
                point_at_obj,
                butil.TRACK_AXIS[rot.get('track_axis', '-z')],
                butil.UP_AXIS[rot.get('up_axis', 'y')])
        else:
            if len(rot) == 3:
                # I think Panda3D's HPR is intrinsic
                # If Blender is extrinsic, the order can just be reversed, LOL
                # rot is in HPR form
                # H -> Z
                # P -> X
                # R -> Y
                # obj.rotation_mode = 'ZXY'
                obj.rotation_mode = 'YXZ'
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


def parse_model(model: Dict) -> ConfigObject:
    """convert dict from json to ModelConfig"""
    model = dict(model)
    children = model.get('children')
    if children is not None:
        children = [parse_model(x) for x in children]
        model['children'] = children

    print(model, flush=True)

    model_class = model['class']
    del model['class']

    if model_class == ConfigTypes.MODEL:
        return ConfigModel(**model)
    elif model_class == ConfigTypes.INSTANCE:
        return ConfigInstance(**model)
    elif model_class == ConfigTypes.EMPTY:
        return ConfigEmpty(**model)
    elif model_class == ConfigTypes.LIGHT:
        return ConfigLight(**model)
    elif model_class == ConfigTypes.ARMATURE:
        return ConfigArmature(**model)
    elif model_class == ConfigTypes.BONE:
        return ConfigBone(**model)
    else:
        print(f'invalid config class:', model_class)
        sys.exit()
