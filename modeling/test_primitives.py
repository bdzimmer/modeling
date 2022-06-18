"""
Tests for primitives.
"""

# Copyright (c) 2022 Ben Zimmer. All rights reserved.

from modeling import util, primitives, ops
from modeling.types import vec3

DEBUG = False


def test_uv_sphere():
    """test uv_sphere"""
    mesh = primitives.uv_sphere(1.0, 8, 8)

    if DEBUG:
        util.view_mesh(mesh)

    # 8, 8
    assert mesh[0].shape[0] == 8 * 7 + 2


def test_uv_capsule():
    """test uv_capsule"""

    mesh_a = primitives.uv_capsule(2.0, 1.0, 8, 3)
    mesh_b = primitives.uv_capsule(2.0, 1.0, 8, 4)
    mesh_c = primitives.uv_capsule(2.0, 1.0, 8, 5)
    mesh_d = primitives.uv_capsule(2.0, 1.0, 8, 6)

    if DEBUG:
        util.view_mesh(
            util.concat([
                ops.translate_mesh(mesh_a, vec3(-4.5, 0, 0)),
                ops.translate_mesh(mesh_b, vec3(-1.5, 0, 0)),
                ops.translate_mesh(mesh_c, vec3(1.5, 0, 0)),
                ops.translate_mesh(mesh_d, vec3(4.5, 0, 0)),
            ])
        )

    # 8, 3
    assert mesh_a[0].shape[0] == 8 * 2 + 8 * 2 + 2

    # 8, 4
    assert mesh_b[0].shape[0] == 8 * 2 + 8 * 2 + 2

    # 8, 5
    assert mesh_c[0].shape[0] == 8 * 4 + 8 * 2 + 2

    # 8, 6
    assert mesh_d[0].shape[0] == 8 * 4 + 8 * 2 + 2


def test_cylinder():
    """test cylinder"""
    subdivisions = 16
    mesh = primitives.cylinder(1.0, 1.0, subdivisions)

    if DEBUG:
        util.view_mesh(mesh)

    assert mesh[0].shape[0] == subdivisions * 2 + 2  # two loops plus end points
    assert mesh[1].shape[0] == subdivisions * 2 + subdivisions * 2  # center plus ends
