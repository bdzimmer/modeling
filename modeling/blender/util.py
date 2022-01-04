"""

Blender utilities.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.


from typing import Any, List, Tuple

import bpy
import bpy.types as btypes

DEBUG = True

TRACK_NEGATIVE_Z = 'TRACK_NEGATIVE_Z'

TRACK_AXIS = {
    '-z': TRACK_NEGATIVE_Z
}

UP_X = 'UP_X'
UP_Y = 'UP_Y'

UP_AXIS = {
    'x': UP_X,
    'y': UP_Y
}


def find_args(argv: List[str]) -> List[str]:
    """find relevant args for script from complete command line args"""
    try:
        idx_start = argv.index('--')
    except:
        return list()
    return argv[(idx_start + 1):]


def disable_splash():
    """disable startup splash when running from command line"""
    # https://blender.stackexchange.com/questions/5208/prevent-splash-screen-from-being-shown-when-using-a-script
    bpy.context.preferences.view.show_splash = False


def delete_all_objects():
    """delete all objects in scene / file"""

    # using remove seems to avoid segfaults
    # https://docs.blender.org/api/current/bpy.types.BlendDataObjects.html#bpy.types.BlendDataObjects.remove

    def clear(obj_list):
        """helper to show what we're clearing"""
        for obj in obj_list:
            print("removing", obj)
            obj_list.remove(obj)

    clear(bpy.data.objects)
    clear(bpy.data.materials)
    clear(bpy.data.images)
    clear(bpy.data.cameras)
    clear(bpy.data.lights)
    clear(bpy.data.meshes)


def purge_orphans():
    """purge orphan data-blocks"""

    # https://blender.stackexchange.com/questions/121531/how-do-i-delete-unused-data

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def reset_scene():
    """reset the scene"""
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.context.scene.cursor.rotation_mode = "QUATERNION"
    bpy.context.scene.cursor.rotation_quaternion = (1, 0, 0, 0)


def select_none() -> None:
    """unselect all objects"""
    for obj in bpy.data.objects:
        obj.select_set(False)


def select(obj: btypes.Object) -> None:
    """select an object by reference"""
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def hide(obj: btypes.Object) -> None:
    """hide an object from render"""
    # obj.hide_viewport = True
    obj.hide_render = True


def show(obj: btypes.Object) -> None:
    """show an object for render"""
    # obj.hide_viewport = False
    obj.hide_render = False


def quit():
    """quit blender"""
    bpy.ops.wm.quit_blender()


def get_obj_by_name(name: str) -> bpy.types.Object:
    """get an object by name"""
    return bpy.context.scene.objects.get(name)


def add_subsurface(
        obj: btypes.Object,
        levels: int,
        render_levels: int,
        use_adaptive_subdivision: bool) -> None:
    """add a subsurface modifier"""
    sub = obj.modifiers.new('Subsurface', 'SUBSURF')
    sub.levels = levels
    sub.render_levels = render_levels
    obj.cycles.use_adaptive_subdivision = use_adaptive_subdivision


def add_link(mat, conn_out, conn_in):
    """add a link between an output of one shader and an input of another"""

    if DEBUG:
        print(
            conn_out[0].__class__,
            conn_out[1],
            "->",
            conn_in[0].__class__,
            conn_in[1]
        )

    mat.node_tree.links.new(
        conn_out[0].outputs[conn_out[1]],
        conn_in[0].inputs[conn_in[1]])


def arrange_nodes(columns):
    """arrange nodes by columns"""
    # for this to work, a shader view has to be open

    # I forget the purpose of this
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

    x_gap = 64
    y_gap = 16
    x_pos = 0.

    for column in columns:
        y_pos = 0.0
        if isinstance(column, list):
            max_width = max([x.dimensions[0] for x in column])
            for node in column:
                node.location = (x_pos, -y_pos)
                y_pos = y_pos + node.dimensions[1] + y_gap
            x_pos = x_pos + max_width + x_gap
        elif column is None:
            x_pos = x_pos + x_gap * 4
        else:
            node = column
            node.location = (x_pos, -y_pos)
            x_pos = x_pos + node.dimensions[0] + x_gap
            # x_pos = x_pos + max_width + x_gap


def arrange_nodes_hierarchy(columns: List[Any], x_pos: float, y_pos: float) -> None:
    """a little bit more powerful version"""
    print("arranging at offset", x_pos, y_pos)

    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    x_gap = 64
    y_gap = 16
    y_chunk_gap = 64
    y_pos_org = y_pos

    for column in columns:
        y_pos = y_pos_org

        if isinstance(column, list):
            max_width = 0

            for node in column:
                if isinstance(node, tuple):
                    arrange_nodes_hierarchy(list(node), x_pos, y_pos)
                    stuff_flat = flatten_nodes(node)  # find dimensions of chunk we just arranged
                    _, dim = dimensions(stuff_flat)
                    # x_pos = x_pos + dim[0] + x_gap
                    y_pos = y_pos + dim[1] + y_chunk_gap
                    if dim[0] > max_width:
                        max_width = dim[0]
                else:
                    node.location = (x_pos, -y_pos)
                    y_pos = y_pos + node.dimensions[1] + y_gap
                    if node.dimensions[0] > max_width:
                        max_width = node.dimensions[0]

            x_pos = x_pos + max_width + x_gap

        else:

            # TODO: potentially handle the tuple case here too

            node = column
            node.location = (x_pos, -y_pos)
            x_pos = x_pos + node.dimensions[0] + x_gap


def dimensions(nodes: List) -> Tuple:
    """calculate dimensions of a set of nodes"""

    x_vals = [x.location[0] for x in nodes] + [x.location[0] + x.dimensions[0] for x in nodes]
    y_vals = [x.location[1] for x in nodes] + [x.location[1] + x.dimensions[1] for x in nodes]

    return (
        (min(x_vals), min(y_vals)),
        (max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)))


def flatten_nodes(nodes: Any) -> List:
    """recursively flatten any nested arrangement of nodes to a list"""

    # TODO: fix this; the base case is not simple enough but I can't think

    if not isinstance(nodes, list) and not isinstance(nodes, tuple):
        return [nodes]

    res = []

    for thing in nodes:
        if not isinstance(thing, list) and not isinstance(thing, tuple):
            res.append(thing)
        else:
            res.extend(
                [
                    y
                    for x in thing
                    for y in flatten_nodes(x)
                ]
            )

    return res


def sun(
        name: str,
        loc: Tuple,
        rot_euler: Tuple,
        energy: float,
        angle: float):
    """build a sun"""
    # pylint: disable=too-many-arguments

    bpy.ops.object.light_add()

    obj = bpy.context.object

    obj.name = name
    obj.location = loc
    obj.rotation_euler = rot_euler

    obj.data.name = name
    obj.data.type = 'SUN'
    obj.data.energy = energy
    obj.data.angle = angle

    return obj


def point_at(
        src: btypes.Object,
        dst: btypes.Object,
        track_axis: str,
        up_axis: str) -> None:
    """point one object at another using a constraint"""
    track_to = src.constraints.new('TRACK_TO')
    track_to.target = dst
    track_to.track_axis = track_axis
    track_to.up_axis = up_axis


def configure_cycles(
        scene: btypes.Scene,
        samples: int,
        preview_samples: int,
        gpu: bool) -> None:
    """set commonly used settings for cycles"""

    scene.render.engine = "CYCLES"

    # this could be made an argument someday if necessary
    scene.cycles.feature_set = "EXPERIMENTAL"

    scene.cycles.samples = samples
    scene.cycles.preview_samples = preview_samples

    if gpu:
        scene.cycles.device = 'GPU'


def render_animation(
        scene: btypes.Scene,
        output_dirname: str,
        frame_start: int, frame_end: int) -> None:
    """render animation frames to a directory"""

    # trailing slash is required here
    scene.render.filepath = output_dirname + '/'

    scene.frame_start = frame_start
    scene.frame_end = frame_end

    bpy.ops.render.render(animation=True)


def render(output_filename: str) -> None:
    """render a still image"""

    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(filepath=output_filename)

