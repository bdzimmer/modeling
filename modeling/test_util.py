"""

Unit tests for modeling utilities.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from modeling import util, primitives
from modeling.types import vec3


def test_union():
    """unit test for wrapper around trimesh union"""

    # all of this works with trimesh 3.2.2

    # no redundant points
    mesh = primitives.uv_sphere(4.0, 8, 8)
    mesh_2 = util.translate_mesh(mesh, vec3(0.5, 0, 0))

    # concat / union meshes together
    mesh_c = util.concat([mesh, mesh_2])
    mesh_u = util.union([mesh, mesh_2])

    # union the mesh with just itself
    mesh_u_self = util.union([mesh])

    # util.view_mesh(mesh)
    # util.view_mesh(mesh_c)
    # util.view_mesh(mesh_u)
    # util.view_mesh(mesh_u_self)

    # TODO: add assertions
