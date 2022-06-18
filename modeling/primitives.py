"""
Various primitives, including wrappers around trimesh primitives.
"""

# Copyright (c) 2022 Ben Zimmer. All rights reserved.
import trimesh
from typing import Tuple

import numpy as np
import trimesh as tm

from modeling.types import Mesh, vec3
from modeling import util, solid, ops


def icosphere(subdivisions: int, radius: float) -> Mesh:
    """icosphere"""
    mesh_tm = tm.primitives.creation.icosphere(subdivisions, radius, None)
    return mesh_tm.vertices, mesh_tm.faces


def uv_sphere_trimesh(radius: float, count: Tuple[int, int]) -> Mesh:
    """uv sphere from trimesh...don't use"""
    mesh_tm = tm.primitives.creation.uv_sphere(radius, count)
    return mesh_tm.vertices, mesh_tm.faces


def uv_sphere(radius: float, segs: int, rings: int) -> Mesh:
    """UV sphere"""

    yx = [
        util.circle_point(x, radius)
        for x in np.linspace(0, ops.TAU / 2, rings + 1)[1:-1]
    ]

    ribs = []
    for y, x in yx:
        rib = np.array(util.points_around_circle(segs, 0.0, x))
        rib = util.attach_y(rib, y)
        ribs.append(rib)

    return solid.solid_from_ribs(
        ribs,
        vec3(0, radius, 0),
        vec3(0, -radius, 0)
    )


def uv_capsule(
        cyl_length: float,
        radius: float,
        segs: int,
        rings: int) -> Mesh:
    """cylinder with a sphere stuck on each end"""

    yx = [
        util.circle_point(x, radius)
        for x in np.linspace(0, ops.TAU / 2, rings + 1)[1:-1]
    ]

    center_ring = np.array(util.points_around_circle(segs, 0.0, radius))

    ribs = []
    for idx, (y, x) in enumerate(yx):

        rib = np.array(util.points_around_circle(segs, 0.0, x))

        if idx < (rings - 1) // 2:
            y = y + cyl_length / 2
        else:
            y = y - cyl_length / 2

        # print(idx, ':', y, flush=True)

        rib = util.attach_y(rib, y)

        ribs.append(rib)

        # if there is an even number of sphere ribs,
        # we need to add two center ribs in a certain spot
        if (rings - 1) % 2 == 0 and idx == (int(rings - 1) // 2) - 1:
            # print('got here', idx)
            # print('extra :', cyl_length / 2, flush=True)
            rib = util.attach_y(center_ring, cyl_length / 2)
            ribs.append(rib)
            # print('extra :', -cyl_length / 2, flush=True)
            rib = util.attach_y(center_ring, -cyl_length / 2)
            ribs.append(rib)

        # if there is an odd number of sphere ribs,
        # we only need to add one additional center rib
        if (rings - 1) % 2 == 1 and idx == (int(rings - 1) // 2) - 1:
            # print('got here', idx)
            # print('extra :', cyl_length / 2, flush=True)
            rib = util.attach_y(center_ring, cyl_length / 2)
            ribs.append(rib)

    return solid.solid_from_ribs(
        ribs,
        vec3(0, radius + cyl_length / 2, 0),
        vec3(0, -radius - cyl_length / 2, 0)
    )


def cylinder(length: float, radius: float, subdivisions: int) -> Mesh:
    """cylinder with closed caps"""

    half_length = length / 2.0

    outline = np.array([
        [radius, -half_length],
        [radius, half_length]
    ])

    curve_verts, curve_faces = solid.surface_revolution(
        outline, np.pi * 2.0, True, False, subdivisions)

    center_0 = np.array([[0.0, -half_length, 0.0]])
    center_1 = np.array([[0.0, half_length, 0.0]])

    center_0_idx = curve_verts.shape[0]
    center_1_idx = curve_verts.shape[0] + 1

    cap_0_idxs = range(0, subdivisions * 2, 2)
    cap_1_idxs = range(1, subdivisions * 2, 2)

    cap_0_faces = solid.close_face_loop(cap_0_idxs, center_0_idx)
    cap_1_faces = solid.close_face_loop(cap_1_idxs[::-1], center_1_idx)

    return (
        np.concatenate((curve_verts, center_0, center_1), axis=0),
        np.concatenate((curve_faces, cap_0_faces, cap_1_faces), axis=0))


def box_mesh(x_extent: float, y_extent: float, z_extent: float) -> Mesh:
    """create a box mesh"""
    # wrapper around trimesh interface
    # TODO: my own implementation of this would be nice
    box = trimesh.primitives.Box(extents=(x_extent, y_extent, z_extent)).to_mesh()
    return box.vertices, box.faces