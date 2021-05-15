"""

Write meshes to Panda3D egg format.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from typing import Optional

import numpy as np

from panda3d import core
from panda3d import egg


def write(
        filename: str,
        egg_data: egg.EggData) -> None:
    """write egg data to a file"""

    egg_data.writeEgg(core.Filename(filename))


def mesh_to_egg(
        verts: np.ndarray,
        faces: np.ndarray,
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

    for vert_np in verts:
        vert_egg = egg.EggVertex()
        vert_egg.setPos(core.Point3D(*vert_np))
        vp.addVertex(vert_egg)

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

    return data
