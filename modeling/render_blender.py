"""

Use Blender to render a model to a PNG.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.
import os

import json
import sys
from typing import Any, Dict, Optional
import math

import bpy


# TODO: there is probably a way to find this at runtime
CODE_DIRNAME = '/home/ben/code/modeling'

if CODE_DIRNAME not in sys.path:
    sys.path.append(CODE_DIRNAME)

from modeling import blender

DO_RENDER = True
DO_QUIT = True


CYCLES_RENDER_SAMPLES = 8
CYCLES_PREVIEW_SAMPLES = 8


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
    output_filename_prefix = os.path.splitext(output_filename)[0]

    print('input filename: ', input_filename)
    print('output filename:', output_filename)

    # ~~~~ load input json

    with open(input_filename, 'r') as json_file:
        config = json.load(json_file)

    do_outline = config.get('do_outline', False)

    # a couple of hard-coded things for now
    clip_scale = 4.0
    sun_energy = config.get('sun_energy', 25.0)

    # some hard-coded stuff for working with the first model

    model_config = config['models'][0]

    scale = model_config.get('pos_scale', 1.5)
    center = model_config['center']

    pos = model_config['size'] * scale
    clip_end = model_config['size'] * clip_scale

    root_obj_loc = (-center[0], -center[1], -center[2])
    cam_pos = (pos * 0.75, pos * 1.5, pos * 0.5)
    sun_pos = (pos, pos * 0.25, pos * 0.3)

    # ~~~~ clear scene

    blender.delete_all_objects()
    blender.reset_scene()

    # ~~~~ load OBJ file

    # for now, just load starting from the first model
    root_obj = add_model(config['models'][0], None)

    # TODO: do this with some kind of offset instead
    root_obj.location = root_obj_loc

    # ~~~~ origin object

    origin_obj = bpy.data.objects.new('origin', None)
    bpy.context.scene.collection.objects.link(origin_obj)
    origin_obj.location = (0, 0, 0)

    # ~~~~ camera

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = "Camera"
    cam.location = cam_pos
    cam.data.clip_end = clip_end

    # blender.point_at(cam, root_obj, 'TRACK_NEGATIVE_Z', 'UP_Y')
    blender.point_at(cam, origin_obj, 'TRACK_NEGATIVE_Z', 'UP_Y')

    bpy.context.scene.camera = cam

    # ~~~~ Light

    sun = blender.sun(
        name='Sun',
        loc=sun_pos,
        rot_euler=(0.0, 0.0, 0.0),
        energy=sun_energy,
        angle=0.0,  # 10.0 * math.pi / 180.0,
    )
    blender.point_at(sun, root_obj, 'TRACK_NEGATIVE_Z', 'UP_Y')

    # ~~~~ render settings

    scene = bpy.context.scene

    # set background color

    # background = scene.world.node_tree.nodes['Background']
    # background.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.render.film_transparent = True

    scene.render.engine = 'CYCLES'
    blender.configure_cycles(
        scene,
        samples=CYCLES_RENDER_SAMPLES,
        preview_samples=CYCLES_PREVIEW_SAMPLES)

    # bpy.context.scene.render.filepath = working_dirname + '/'
    # bpy.context.scene.cycles.use_denoising = True
    scene.view_settings.look = 'Medium High Contrast'

    if DO_RENDER:

        # standard render
        render(output_filename)

        if do_outline:
            # WIP: outline mode for schematics

            # TODO: disable rendering everything except freestyle
            scene.render.engine = 'BLENDER_EEVEE'

            set_render_outlines(scene, line_thickness=1.0)
            cam.data.type = 'ORTHO'
            cam.data.clip_start = 0
            cam.data.clip_end = pos * 2.0
            cam.data.ortho_scale = pos * 1.2

            # root_obj.location = (0, 0, 0)

            for name, cam_pos, up_dir in [
                    ('pos_x', (1, 0, 0), 'UP_Y'),
                    ('pos_y', (0, 1, 0), 'UP_Y'),
                    ('neg_y', (0, -1, 0), 'UP_Y'),
                    ('pos_z', (0, 0, 1), 'UP_X'),
                    ('neg_z', (0, 0, -1), 'UP_X')]:
                print(name)
                cam_pos = (
                    cam_pos[0] * pos,
                    cam_pos[1] * pos,
                    cam_pos[2] * pos)
                cam.location = cam_pos
                blender.point_at(cam, origin_obj, 'TRACK_NEGATIVE_Z', up_dir)
                render(output_filename_prefix + '_outline_' + name + '.png')

    if DO_QUIT:
        blender.quit()


def set_render_outlines(scene: bpy.types.Scene, line_thickness: float) -> None:
    """set up a scene for rendering outlines using freestyle"""
    scene.use_nodes = True
    scene.render.use_freestyle = True
    scene.render.line_thickness = line_thickness

    # TODO: other relevant freestyle settings
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

    obj = import_obj(model_config['filename'])
    name = model_config['name']
    obj.name = name

    if parent is not None:
        obj.parent = parent
        # TODO: is there another step required to make transforms relative???

    transformation = model_config.get('transformation')
    if transformation is not None:
        # for now just translation. rotation later
        obj.location = transformation['translation']

    # enable smooth shading
    auto_smooth_angle = model_config.get('auto_smooth_angle')
    if auto_smooth_angle is not None:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = auto_smooth_angle * math.pi / 180.0
        bpy.ops.object.shade_smooth()

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


main(blender.find_args(sys.argv))
