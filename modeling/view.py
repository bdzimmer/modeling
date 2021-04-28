"""

Interactive 3D viewer.

"""

# Copyright (c) 2019 Ben Zimmer. All rights reserved.

import trimesh
import pyrender


def view(mesh, material):
    """view a single mesh with nice default settings"""
    scene = pyrender.Scene()
    if isinstance(mesh, tuple):
        mesh = trimesh.Trimesh(*mesh)
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
    pyrender.Viewer(
        scene,
        viewport_size=(1280, 720),
        render_flags={
            "cull_faces": False},
        viewer_flags={
            "show_world_axis": True,
            "use_raymond_lighting": True,
            "use_perspective_cam": True})
