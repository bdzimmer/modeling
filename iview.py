"""

Interactive viewer for procedural modeling.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

import os
import pickle
import time

import pyrender
import trimesh


MESH_FILENAME = 'models/cockpit2.obj'
COLORS_FILENAME = None

MESH_MATERIAL = pyrender.material.MetallicRoughnessMaterial(
    baseColorFactor=(180, 180, 190),
    metallicFactor=0.25,
    roughnessFactor=0.5)

BG_COLOR = (0, 0.5, 0.5)


def get_mesh() -> pyrender.Mesh:
    """get a mesh to render"""
    mesh = trimesh.load(MESH_FILENAME)

    colors_filename = os.path.splitext(MESH_FILENAME)[0] + '_fc.pkl'

    if os.path.exists(colors_filename):
        with open(colors_filename, 'rb') as pickle_file:
            face_colors = pickle.load(pickle_file)
        mesh.visual.face_colors = face_colors

    return pyrender.Mesh.from_trimesh(mesh, material=MESH_MATERIAL, smooth=False)


def main():
    """main program"""

    scene = pyrender.Scene(bg_color=BG_COLOR)
    mesh = get_mesh()
    mesh_node = scene.add(mesh)

    viewer = pyrender.Viewer(
        scene,
        viewport_size=(1280, 720),
        render_flags={
            "cull_faces": False},
        viewer_flags={
            # "show_world_axis": True,
            "use_raymond_lighting": True,
            "use_perspective_cam": True
        },
        run_in_thread=True)

    while True:

        viewer.render_lock.acquire()

        if mesh_node is not None:
            scene.remove_node(mesh_node)

        mesh = get_mesh()
        mesh_node = scene.add(mesh)

        viewer.render_lock.release()

        time.sleep(5.0)


if __name__ == '__main__':
    main()
