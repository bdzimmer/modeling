"""

Use Blender to render a scene to one or more image files.

"""

# Copyright (c) 2022 Ben Zimmer. All rights reserved.

import math
import pickle
import os
import json
import sys
from typing import Optional

import bpy
import bpy.types as btypes


CODE_DIR_PATH = os.path.dirname(os.path.dirname(__file__))

if CODE_DIR_PATH not in sys.path:
    sys.path.append(CODE_DIR_PATH)

from modeling.blender import util as butil, scene as msc, materials as ms
from modeling.blender.scene import parse_model
from modeling.blender.types import MaterialKeys as Mk
from modeling.blender import types


DO_RENDER = True

# TODO: make these options
# CYCLES_RENDER_SAMPLES = 128
CYCLES_RENDER_SAMPLES = 32

# CYCLES_PREVIEW_SAMPLES = 32
CYCLES_PREVIEW_SAMPLES = 4

LIGHT_SUN = 'sun'


# keys, mainly useed for the config render portion of the input config
class ConfigKeys:
    SIZE_KEY = 'size'
    CENTER_KEY = 'center'
    POS_SCALE_KEY = 'pos_scale'
    FILM_TRANSPARENT_KEY = 'film_transparent'
    LINE_THICKNESS_KEY = 'line_thickness'
    ORTHO_SCALE_KEY = 'ortho_scale'
    RENDER_EEVEE_USE_BLOOM_KEY = 'render_eevee_use_bloom'
    RENDER_USE_EEVEE_KEY = 'render_use_eevee'
    CYCLES_GPU_KEY = 'cycles_gpu'
    ANIMATION_USE_EEVEE_KEY = 'animation_use_eevee'
    RENDER_RESOLUTION_KEY = 'render_resolution'
    ROOT_OFFSET_KEY = 'root_offset'
    DO_QUIT_KEY = 'do_quit'
    DO_OUTLINE_KEY = 'do_outline'
    DO_RENDER_ANIMATION_KEY = 'do_render_animation'
    DO_RENDER_KEY = 'do_render'
    RENDER_BLENDER_KEY = 'render_blender'
    WORLD_KEY = 'world'


def main(args):
    """main program"""

    # not sure if this does anything
    butil.disable_splash()

    input_filename = args[0]
    output_filename = args[1]

    if len(args) > 2:
        animation_filename = args[2]
    else:
        animation_filename = None

    output_filename_prefix = os.path.splitext(output_filename)[0]

    print('input filename: ', input_filename)
    print('output filename:', output_filename)
    print('animation input filename:', animation_filename)

    # ~~~~ load input json

    with open(input_filename, 'r') as json_file:
        config = json.load(json_file)

    # a couple of hard-coded things for now
    clip_scale = 4.0

    # some render_blender specific settings

    Ck = ConfigKeys
    config_render = config[Ck.RENDER_BLENDER_KEY]

    do_render = config_render.get(Ck.DO_RENDER_KEY, True)
    do_render_animation = config_render.get(Ck.DO_RENDER_ANIMATION_KEY, False)
    do_outline = config_render.get(Ck.DO_OUTLINE_KEY, False)
    do_quit = config_render.get(Ck.DO_QUIT_KEY, True)
    root_offset = config_render.get(Ck.ROOT_OFFSET_KEY, False)

    render_resolution = config_render.get(Ck.RENDER_RESOLUTION_KEY, [1920, 1080])
    animation_use_eevee = config_render.get(Ck.ANIMATION_USE_EEVEE_KEY, False)
    render_use_eevee = config_render.get(Ck.RENDER_USE_EEVEE_KEY, False)
    render_eevee_use_bloom = config_render.get(Ck.RENDER_EEVEE_USE_BLOOM_KEY, False)
    cycles_gpu = config_render.get(Ck.CYCLES_GPU_KEY, False)

    ortho_scale = config_render.get(Ck.ORTHO_SCALE_KEY, 1.1)
    line_thickness = config_render.get(Ck.LINE_THICKNESS_KEY, 1.0)
    film_transparent = config_render.get(Ck.FILM_TRANSPARENT_KEY, True)

    scale = config_render.get(Ck.POS_SCALE_KEY, 1.5)
    center = config_render[Ck.CENTER_KEY]
    pos = config_render[Ck.SIZE_KEY] * scale
    clip_end = config_render[Ck.SIZE_KEY] * clip_scale

    world_config = config_render.get(Ck.WORLD_KEY, {})

    root_obj_loc = (-center[0], -center[1], -center[2])

    # ~~~~ set some blender defaults

    view_space: btypes.SpaceView3D = find_space('VIEW_3D')
    view_space.shading.show_cavity = True

    bpy.context.scene.transform_orientation_slots[1].type = 'LOCAL'
    view_space.show_gizmo_object_translate = True

    # ~~~~ clear scene

    butil.delete_all_objects()
    butil.reset_scene()

    # ~~~~ create view layers

    scene = bpy.context.scene

    layers = config.get('layers')
    if layers is not None:
        for layer_name in layers:
            if scene.view_layers.get(layer_name) is None:
                bpy.ops.scene.view_layer_add(type='NEW')
                bpy.context.view_layer.name = layer_name
        if layers:
            bpy.context.window.view_layer = scene.view_layers[layers[0]]
            scene.view_layers[-1].use_pass_environment = True

        # set up appropriate compositing
        add_compositor_nodes(scene)

    # ~~~~ create collections

    collections = config.get('collections')
    if collections is not None:
        for col_dict in collections:
            col_name = col_dict['name']
            print(col_name)
            if scene.collection.children.get(col_name) is None:
                collection = bpy.data.collections.new(col_name)
                scene.collection.children.link(collection)
            for vl in scene.view_layers:
                for col in vl.layer_collection.children:
                    if col.name == col_name and vl.name in col_dict['hide']:
                        print(f'\thiding collection `{col.name}` from view layer `{vl.name}`')
                        col.exclude = True

    # ~~~~ create materials

    materials = config.get('materials')
    if materials is not None:
        if materials:
            parent = msc.add_model(
                types.ConfigEmpty(name='MATERIAL PREVIEWS', hide=True), None)
            for material in materials:

                # if there is no name field, it's expected that we
                # are loading the material by name from the materials library
                if Mk.NAME in material:
                    material_name = material[Mk.NAME]
                else:
                    material_name = (
                        material[Mk.MATLIB][Mk.MATLIB_LIB_NAME] + ' / ' +
                        material[Mk.MATLIB][Mk.MATLIB_MAT_NAME]
                    )

                msc.add_model(
                    types.ConfigModel(
                        name=('MATERIAL PREVIEW - ' + material_name),
                        filename='models/sphere.obj',
                        auto_smooth_angle=30.0,
                        material=material,
                        props=[
                            {
                                'type': 'blender:subsurface',
                                'levels': 2,
                                'render_levels': 4,
                                'use_adaptive_subdivision': False
                            }
                        ],
                        hide=True
                    ),
                    parent)

    # ~~~~ special origin object

    origin_obj = bpy.data.objects.new('origin', None)
    bpy.context.scene.collection.objects.link(origin_obj)
    origin_obj.location = (0, 0, 0)

    # ~~~~ camera

    # for now assume one camera

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = 'Camera'
    cam.data.clip_end = clip_end

    fov = config['camera'].get('fov')
    if fov is not None:
        cam.data.lens_unit = 'FOV'
        cam.data.angle = math.radians(fov)

    bpy.context.scene.camera = cam

    # ~~~~ load models

    msc.PROFILER.tick('all loading')

    # for now, test
    model_trees = [parse_model(x) for x in config['models']]

    # for now, just load starting from the first model
    root_obj = msc.add_model(model_trees[0], None)
    if len(config['models']) > 1:
        for model in model_trees[1:]:
            msc.add_model(model, None)

    msc.PROFILER.tock('all loading')

    # blender.purge_orphans()

    # apply offset from center in configuration
    if root_offset:
        root_obj.location = root_obj_loc

    # ~~~~ set camera transform

    msc.set_transformation(cam, config['camera']['transformation'])

    # ~~~~ lights

    for light_name, light in config['lights'].items():
        if light['type'] == LIGHT_SUN:
            light_obj = butil.sun(
                name=light_name,
                loc=(0.0, 0.0, 0.0),
                rot_euler=(0.0, 0.0, 0.0),
                energy=light['energy'],
                angle=light['angle']
            )
            shadow_cascade_max_distance = light.get('shadow_cascade_max_distance')
            if shadow_cascade_max_distance is not None:
                light_obj.data.shadow_cascade_max_distance = shadow_cascade_max_distance
        else:
            print(f'Invalid light type')

        msc.set_transformation(light_obj, light['transformation'])

        collection_name = light.get(types.ModelKeys.COLLECTION, msc.DEFAULT_COLLECTION)
        if collection_name != msc.DEFAULT_COLLECTION:
            for coll in light_obj.users_collection:
                coll.objects.unlink(light_obj)
            bpy.data.collections[collection_name].objects.link(light_obj)

    # ~~~~ render settings

    # set background color
    background = scene.world.node_tree.nodes['Background']
    background.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)

    scene.render.film_transparent = film_transparent

    scene.render.engine = 'CYCLES'
    butil.configure_cycles(
        scene,
        samples=CYCLES_RENDER_SAMPLES,
        preview_samples=CYCLES_PREVIEW_SAMPLES,
        gpu=cycles_gpu)
    scene.eevee.use_bloom = render_eevee_use_bloom

    scene.render.resolution_x = render_resolution[0]
    scene.render.resolution_y = render_resolution[1]

    # bpy.context.scene.render.filepath = working_dirname + '/'
    # bpy.context.scene.cycles.use_denoising = True
    scene.view_settings.look = 'Medium High Contrast'

    if world_config:

        # bpy.context.space_data.shader_type = 'WORLD'

        add_node = ms.build_add_node(scene.world)

        tex_env = add_node(btypes.ShaderNodeTexEnvironment)
        tex_env.image = bpy.data.images.load(world_config['tex_environment_filepath'])
        tex_env.interpolation = 'Cubic'

        hsv = add_node(btypes.ShaderNodeHueSaturation)
        hsv.inputs['Hue'].default_value = world_config.get('hue', 0.5)
        hsv.inputs['Saturation'].default_value = world_config.get('saturation', 1.0)
        hsv.inputs['Value'].default_value = world_config.get('value', 1.0)

        bright_contrast = add_node(btypes.ShaderNodeBrightContrast)
        bright_contrast.inputs['Bright'].default_value = world_config.get('brightness', 0.0)
        bright_contrast.inputs['Contrast'].default_value = world_config.get('contrast', 0.0)

        world_output = scene.world.node_tree.nodes['World Output']

        links = [
            ((tex_env, 'Color'), (hsv, 'Color')),
            ((hsv, 'Color'), (bright_contrast, 'Color')),
            ((bright_contrast, 'Color'), (background, 'Color'))
        ]
        for link in links:
            butil.add_link(scene.world, *link)

        try:
            butil.arrange_nodes([
                add_node,
                tex_env,
                bright_contrast,
                background,
                world_output
            ])
        except:
            # do nothing
            print('problem while arranging world nodes')

        # bpy.context.space_data.shader_type = 'OBJECT'

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
        scene.render.fps = 60
        scene.frame_start = 0
        scene.frame_end = len(states) - 1

        for config_model in config['models']:
            obj = butil.get_obj_by_name(config_model['name'])
            obj.rotation_mode = 'QUATERNION'

        for frame, state in enumerate(states):
            # print(frame, state)
            print(frame, '/', len(states), flush=True)

            scene.frame_set(frame)

            for name, entity in state['objects'].items():
                # print('\t' + name)
                obj = butil.get_obj_by_name(name)
                if obj is not None:
                    # print('\t\t', entity)
                    msc.add_keyframes(
                        obj,
                        entity.get('transformation'),
                        entity.get('nodes'),
                        entity.get('hide'))
                    # TODO: additional beam specific stuff would be handled here
                else:
                    print('\t\tobject not found')

        scene.frame_set(0)

        for config_model in config['models']:
            name = config_model['name']
            obj = butil.get_obj_by_name(name)
            anim_data = obj.animation_data
            if anim_data is not None:
                action = anim_data.action
                if action is not None:
                    for fcurve in action.fcurves:
                        for keyframe in fcurve.keyframe_points:
                            keyframe.interpolation = 'CONSTANT'
                else:
                    print(f'object {name} has no action')
            else:
                print(f'object {name} has no animation data')

        if do_render_animation:
            if animation_use_eevee:
                scene.render.engine = 'BLENDER_EEVEE'
            scene.render.filepath = os.path.join(output_filename_prefix, 'frames') + '/'
            bpy.ops.render.render(animation=True)

            # reset to cycles
            scene.render.engine = 'CYCLES'

    if do_render:

        # standard render
        if render_use_eevee:
            scene.render.engine = 'BLENDER_EEVEE'

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
                    ('pos_x', (1, 0, 0), butil.UP_Y),
                    ('pos_y', (0, 1, 0), butil.UP_Y),
                    ('neg_y', (0, -1, 0), butil.UP_Y),
                    ('pos_z', (0, 0, 1), butil.UP_X),
                    ('neg_z', (0, 0, -1), butil.UP_X)]:
                print(name)
                cam_pos = (
                    cam_pos[0] * pos,
                    cam_pos[1] * pos,
                    cam_pos[2] * pos)
                cam.location = cam_pos
                butil.point_at(
                    cam, origin_obj, butil.TRACK_NEGATIVE_Z, up_dir)
                render(output_filename_prefix + '_outline_' + name + '.png')

    if do_quit:
        butil.quit()

    msc.PROFILER.summary()
    print('', flush=True)


def set_render_outlines(scene: bpy.types.Scene, line_thickness: float) -> None:
    """set up a scene for rendering outlines using freestyle"""

    scene.use_nodes = True
    scene.render.use_freestyle = True
    scene.render.line_thickness = line_thickness
    scene.view_layers[msc.DEFAULT_VIEW_LAYER].freestyle_settings.as_render_pass = True

    butil.add_link(
        scene,
        (scene.node_tree.nodes['Render Layers'], 'Freestyle'),
        (scene.node_tree.nodes['Composite'], 'Image'))


def render(output_filename: str) -> None:
    """render to an output file"""
    print('rendering...', end='', flush=True)
    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(filepath=output_filename)
    print('done')


def find_space(space_type: str) -> Optional[btypes.Space]:
    """find the first space of a given type"""
    # see here for a list of types:
    # https://docs.blender.org/api/current/bpy.types.Space.html#bpy.types.Space
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == space_type:
                for space in area.spaces:
                    if space.type == space_type:
                        return space
    return None


def add_compositor_nodes(scene: bpy.types.Scene):
    """set up compositor nodes"""

    scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    tree.nodes.clear()

    add_node = ms.build_add_node(scene)

    nodes = []
    links = []

    for layer in scene.view_layers:
        # TODO: layer input
        ln = add_node(btypes.CompositorNodeRLayers)
        ln.name = layer.name
        ln.label = layer.name
        ln.layer = layer.name
        ao = add_node(btypes.CompositorNodeAlphaOver)

        nodes.append((ln, ao))

    comp = add_node(btypes.CompositorNodeComposite)
    view = add_node(btypes.CompositorNodeViewer)

    for idx, (fst, snd) in enumerate(zip(nodes[:-1], nodes[1:])):
        ln_0, ao_0 = fst
        ln_1, ao_1 = snd

        if idx == 0:
            links.append(((ln_0, 'Image'), (ao_1, 2)))
        else:
            links.append(((ao_0, 'Image'), (ao_1, 2)))

        if idx < len(nodes) - 2:
            links.append(((ln_1, 'Image'), (ao_1, 1)))
        else:
            # use environment past for last
            links.append(((ln_1, 'Env'), (ao_1, 1)))

    links.append(((nodes[-1][1], 'Image'), (comp, 'Image')))
    links.append(((nodes[-1][1], 'Image'), (view, 'Image')))

    ms.add_links(
        scene,
        links
    )


main(butil.find_args(sys.argv))
