"""

Use Blender to render a scene to one or more image files.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.
import pickle

import os
import json
import sys
from typing import Any, Dict, Optional
import math

import bpy

# TODO: there is probably a way to find the script directory at runtime
CODE_DIRNAME = '/home/ben/code/modeling'

if CODE_DIRNAME not in sys.path:
    sys.path.append(CODE_DIRNAME)

from modeling import blender

DO_RENDER = True

# TODO: figure out how to control the name of the default collection
DEFAULT_COLLECTION = 'Collection'

CYCLES_RENDER_SAMPLES = 8
CYCLES_PREVIEW_SAMPLES = 8

LIGHT_SUN = 'sun'


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


def main(args):
    """main program"""

    blender.disable_splash()

    input_filename = args[0]
    output_filename = args[1]

    if len(args) > 2:
        animation_filename = args[2]
    else:
        animation_filename = None

    output_filename_prefix = os.path.splitext(output_filename)[0]

    print('input filename: ', input_filename)
    print('output filename:', output_filename)
    print('animation filename:', animation_filename)

    # ~~~~ load input json

    with open(input_filename, 'r') as json_file:
        config = json.load(json_file)

    # a couple of hard-coded things for now
    clip_scale = 4.0

    # some render_blender specific settings

    config_render = config['render_blender']

    do_render = config_render.get('do_render', True)
    do_outline = config_render.get('do_outline', False)
    do_quit = config_render.get('do_quit', True)
    ortho_scale = config_render.get('ortho_scale', 1.1)
    line_thickness = config_render.get('line_thickness', 1.0)

    scale = config_render.get('pos_scale', 1.5)
    center = config_render['center']
    pos = config_render['size'] * scale
    clip_end = config_render['size'] * clip_scale

    root_obj_loc = (-center[0], -center[1], -center[2])

    # ~~~~ clear scene

    blender.delete_all_objects()
    blender.reset_scene()

    # ~~~~ load OBJ files

    # for now, just load starting from the first model
    root_obj = add_model(config['models'][0], None)
    if len(config['models']) > 1:
        for model in config['models'][1:]:
            add_model(model, None)

    # apply offset from center in configuration
    root_obj.location = root_obj_loc

    # ~~~~ special origin object

    origin_obj = bpy.data.objects.new('origin', None)
    bpy.context.scene.collection.objects.link(origin_obj)
    origin_obj.location = (0, 0, 0)

    # ~~~~ camera

    # for now assume one camera

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = "Camera"
    cam.data.clip_end = clip_end

    set_transformation(cam, config['camera']['transformation'])

    bpy.context.scene.camera = cam

    # ~~~~ lights

    for light_name, light in config['lights'].items():
        if light['type'] == LIGHT_SUN:
            light_obj = blender.sun(
                name=light_name,
                loc=(0.0, 0.0, 0.0),
                rot_euler=(0.0, 0.0, 0.0),
                energy=light['energy'],
                angle=light['angle']
            )
        else:
            print(f'Invalid light type')

        set_transformation(light_obj, light['transformation'])

    # ~~~~ render settings

    scene = bpy.context.scene

    # set background color
    background = scene.world.node_tree.nodes['Background']
    background.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)

    # TODO: make this an option
    scene.render.film_transparent = True

    scene.render.engine = 'CYCLES'
    blender.configure_cycles(
        scene,
        samples=CYCLES_RENDER_SAMPLES,
        preview_samples=CYCLES_PREVIEW_SAMPLES)

    # bpy.context.scene.render.filepath = working_dirname + '/'
    # bpy.context.scene.cycles.use_denoising = True
    scene.view_settings.look = 'Medium High Contrast'

    # ~~~~ animations

    if animation_filename is not None:

        states = []
        with open(animation_filename, 'rb') as animation_file:
            while True:
                try:
                    states.append(pickle.load(animation_file))
                except EOFError:
                    break

        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = len(states)

        for config_model in config['models']:
            obj = blender.get_obj_by_name(config_model['name'])
            obj.rotation_mode = 'QUATERNION'

        for frame, state in enumerate(states):
            print(frame, state)
            for name, entity in state['entities'].items():
                obj = blender.get_obj_by_name(name)

                obj.keyframe_insert('location', frame=frame)
                obj.location = tuple(entity['transformation']['translation'])

                obj.keyframe_insert('rotation_quaternion', frame=frame)
                obj.rotation_quaternion = tuple(entity['transformation']['rotation'])

        for config_model in config['models']:
            obj = blender.get_obj_by_name(config_model['name'])
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'CONSTANT'

    if do_render:

        # standard render
        render(output_filename)

        if do_outline:
            # outline mode for schematics
            # TODO: is there a way disable rendering everything except freestyle?

            scene.render.engine = 'BLENDER_EEVEE'
            scene.render.resolution_x = 1080
            scene.render.resolution_y = 1080

            set_render_outlines(scene, line_thickness=line_thickness)
            cam.data.type = 'ORTHO'
            cam.data.clip_start = 0
            cam.data.clip_end = pos * 2.0
            cam.data.ortho_scale = config_render['size'] * ortho_scale

            # root_obj.location = (0, 0, 0)

            for name, cam_pos, up_dir in [
                    ('pos_x', (1, 0, 0), blender.UP_Y),
                    ('pos_y', (0, 1, 0), blender.UP_Y),
                    ('neg_y', (0, -1, 0), blender.UP_Y),
                    ('pos_z', (0, 0, 1), blender.UP_X),
                    ('neg_z', (0, 0, -1), blender.UP_X)]:
                print(name)
                cam_pos = (
                    cam_pos[0] * pos,
                    cam_pos[1] * pos,
                    cam_pos[2] * pos)
                cam.location = cam_pos
                blender.point_at(
                    cam, origin_obj, blender.TRACK_NEGATIVE_Z, up_dir)
                render(output_filename_prefix + '_outline_' + name + '.png')

    if do_quit:
        blender.quit()


def set_render_outlines(scene: bpy.types.Scene, line_thickness: float) -> None:
    """set up a scene for rendering outlines using freestyle"""

    scene.use_nodes = True
    scene.render.use_freestyle = True
    scene.render.line_thickness = line_thickness
    scene.view_layers['View Layer'].freestyle_settings.as_render_pass = True

    blender.add_link(
        scene,
        (scene.node_tree.nodes['Render Layers'], 'Freestyle'),
        (scene.node_tree.nodes['Composite'], 'Image'))


def render(output_filename: str) -> None:
    """render to an output file"""
    print('rendering...', end='', flush=True)
    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(filepath=output_filename)
    print('done')


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

        color = model_config.get('color')

        # create a new material for the object
        # not sure this is correct
        # material = bpy.data.materials['Default OBJ']
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True

        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        if color is not None:
            material.node_tree.nodes['Principled BSDF'].inputs[0].default_value = color

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


main(blender.find_args(sys.argv))
