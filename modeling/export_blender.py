"""

Automate the export of a Blender project.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

import os
import json
import sys

import bpy
from bpy.app.handlers import persistent

# TODO: there is probably a way to find the script directory at runtime
CODE_DIRNAME = '/home/ben/code/modeling'

if CODE_DIRNAME not in sys.path:
    sys.path.append(CODE_DIRNAME)

from modeling import blender

DO_RENDER = True
DO_QUIT = True


def main(args):
    """main program"""

    blender.disable_splash()

    input_filename = args[0]
    output_filename = args[1]
    # output_filename_prefix = os.path.splitext(output_filename)[0]

    print('input filename: ', input_filename)
    print('output filename:', output_filename)

    # ~~~~ load input json

    with open(input_filename, 'r') as json_file:
        config = json.load(json_file)

    @persistent
    def load_handler(dummy):
        """stuff to run after project is loaded"""

        # select objects
        object_names = config['object_names']
        for object_name in object_names:
            obj = bpy.context.scene.objects.get(object_name)
            if obj is not None:
                obj.select_set(True)

        # export OBJ format

        obj_filename = output_filename + '.obj'
        bpy.ops.export_scene.obj(
            filepath=obj_filename,
            check_existing=False,
            use_selection=True,
            use_normals=False,
            use_triangles=True,
            use_uvs=False,
            axis_forward='Y',
            axis_up='Z')

        if DO_QUIT:
            blender.quit()

    # set up handler to run after the project is loaded
    bpy.app.handlers.load_post.append(load_handler)

    # ~~~~ load project

    bpy.ops.wm.open_mainfile(filepath=config['input_filename'])


main(blender.find_args(sys.argv))
