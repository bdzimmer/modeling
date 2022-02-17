"""

Automate the export of a Blender project.

"""

# Copyright (c) 2022 Ben Zimmer. All rights reserved.

import json
import os
import sys

import bpy
from bpy.app.handlers import persistent

CODE_DIR_PATH = os.path.dirname(os.path.dirname(__file__))

if CODE_DIR_PATH not in sys.path:
    sys.path.append(CODE_DIR_PATH)

from modeling.blender import util as butil

DO_RENDER = True
DO_QUIT = True


def main(args):
    """main program"""

    butil.disable_splash()

    input_filename = args[0]
    output_filename = args[1]

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
            use_edges=False,
            use_uvs=False,
            axis_forward='Y',
            axis_up='Z')

        if DO_QUIT:
            butil.quit()

    # set up handler to run after the project is loaded
    bpy.app.handlers.load_post.append(load_handler)

    # ~~~~ load project

    bpy.ops.wm.open_mainfile(filepath=config['input_filename'])


main(butil.find_args(sys.argv))
