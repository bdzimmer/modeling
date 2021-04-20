"""

Use Blender to render a model to a PNG.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

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
DO_QUIT = False
INTERACTIVE_RENDER = False


CYCLES_RENDER_SAMPLES = 32
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

    # working_dirname = args[0]
    input_filename = args[0]
    output_filename = args[1]

    # print('working dirname:', working_dirname)
    print('input filename: ', input_filename)
    print('output filename:', output_filename)

    # input_filename = os.path.join(working_dirname, input_filename)
    # output_filename = os.path.join(working_dirname, output_filename)

    # ~~~~ load input json

    with open(input_filename, 'r') as json_file:
        config = json.load(json_file)

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
    root_obj.location = root_obj_loc

    # ~~~~ Camera

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = "Camera"
    cam.location = cam_pos
    cam.data.clip_end = clip_end
    blender.point_at(cam, root_obj, 'TRACK_NEGATIVE_Z', 'UP_Y')
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

    # set black background
    background = bpy.context.scene.world.node_tree.nodes['Background']
    background.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)

    blender.configure_cycles(
        bpy.context.scene,
        samples=CYCLES_RENDER_SAMPLES,
        preview_samples=CYCLES_PREVIEW_SAMPLES)

    # bpy.context.scene.render.filepath = working_dirname + '/'
    # bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.view_settings.look = 'Medium High Contrast'

    if DO_RENDER:
        print('rendering...', end='', flush=True)
        if INTERACTIVE_RENDER:
            bpy.ops.render.render("INVOKE_DEFAULT", animation=False, write_still=True)
        else:
            bpy.ops.render.render()
            bpy.data.images["Render Result"].save_render(filepath=output_filename)
        print('done')

    if DO_QUIT:
        blender.quit()


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
