"""

Write meshes to Panda3D egg format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from panda3d import core
from panda3d import egg

from modeling.types import Mesh
from modeling import util


def write(
        filename: str,
        egg_data: egg.EggData) -> None:
    """write egg data to a file"""

    egg_data.writeEgg(core.Filename(filename))


def find_vertex_normals(mesh: Mesh, auto_smooth_angle: float) -> Tuple[Mesh, np.ndarray]:
    """
    create a new mesh and corresponding vertex normals
    given an autosmooth angle
    """

    vert_count_before = mesh[0].shape[0]
    face_count_before = mesh[1].shape[0]

    mesh = util.autosmooth(mesh, auto_smooth_angle)
    mesh_tm = util.tm(*mesh)

    verts = mesh_tm.vertices
    faces = mesh_tm.faces
    vert_normals = mesh_tm.vertex_normals

    vert_count_after = verts.shape[0]
    face_count_after = faces.shape[0]

    print(
        'autosmooth:',
        f'{vert_count_before}, {face_count_before} -> {vert_count_after}, {face_count_after}')

    return (verts, faces), vert_normals


def mesh_to_egg(
        verts: np.ndarray,
        faces: np.ndarray,
        vertex_normals: Optional[np.ndarray],
        vertex_colors: Optional[np.ndarray],
        face_colors: Optional[np.ndarray],
        vertex_pool_name: str) -> egg.EggData:
    """convert a mesh to egg data"""

    # TODO: experiment with different coordinate systems
    z_up = egg.EggCoordinateSystem()
    z_up.setValue(core.CSZupRight)

    data = egg.EggData()
    data.addChild(z_up)

    vp = egg.EggVertexPool(vertex_pool_name)
    data.addChild(vp)

    for idx, vert_np in enumerate(verts):
        vert_egg = egg.EggVertex()
        vert_egg.setPos(core.Point3D(*vert_np))

        if vertex_normals is not None:
            vert_egg.setNormal(core.Point3D(*vertex_normals[idx]))

        if vertex_colors is not None:
            color = core.Vec4(*vertex_colors[idx])
            vert_egg.setColor(color)

        vp.addVertex(vert_egg, idx)

    # TODO: optionally put polygons in multiple EggGroups
    for face_idx, face_np in enumerate(faces):
        poly = egg.EggPolygon()
        for idx in face_np:
            vert = vp.getVertex(int(idx))
            poly.addVertex(vert)
            if face_colors is not None:
                color = core.Vec4(*face_colors[face_idx])
                poly.setColor(color)
        data.addChild(poly)

    # If we don't have vertex normals, we at least want polygon normals.
    if vertex_normals is None:
        data.recomputePolygonNormals()

    return data
