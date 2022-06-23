"""

Functions for creating solids.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from typing import List, Tuple, Optional, Callable, Union

import numpy as np

from modeling import util, ops
from modeling.types import \
    Verts2D, Mesh, Verts, Point3, vec3, Faces, MeshExtended, Vec3, vec
from modeling.util import attach_y


def solid_from_ribs(
        ribs: List[Verts],
        pt_start: Optional[Point3],
        pt_end: Optional[Point3]) -> Mesh:
    """make a solid from a list of 3d ribs and cap y positions"""

    if pt_start is not None:
        mesh = util.point_mesh(pt_start)
        mesh, ex_idxs, _ = extend_from_cap(mesh, 0, ribs[0], True)
    else:
        mesh = ribs[0], np.empty((0, 3), dtype=np.int)
        ex_idxs = list(range(mesh[0].shape[0]))

    for rib in ribs[1:]:
        mesh, ex_idxs, _ = extend_link_loop(mesh, ex_idxs, rib)

    if pt_end is not None:
        mesh, ex_idxs, _ = extend_to_cap(mesh, util.loop(ex_idxs), pt_end)

    return mesh


def solid_from_rib_scale(
        rib_xz: Verts2D,
        y_rad: List[Tuple[float, float]],
        caps: bool) -> Mesh:
    """
    Make a solid by placing a rib at different
    positions and scales.

    Each tuple in y_rad is a position and or
    scale factor or "radius".
    """

    ribs = [attach_y(rib_xz * rad, y) for y, rad in y_rad]

    if caps:
        pt_start, pt_end = _get_cap_points(y_rad)
    else:
        pt_start, pt_end = None, None

    return solid_from_ribs(ribs, pt_start, pt_end)


def solid_from_rib_inset(
        rib_xz: Verts2D,
        y_inset: List[Tuple[float, Optional[float]]],
        caps: bool) -> Mesh:
    """
    Make a solid by placing a rib at different
    positions and insets.

    Each tuple in y_rad is a position and or
    scale factor or "radius".
    """

    ribs = []
    for y, inset in y_inset:
        rib = rib_xz
        if inset is not None:
            rib = util.inset_polygon(rib, inset)
        rib = attach_y(rib, y)
        ribs.append(rib)

    if caps:
        pt_start, pt_end = _get_cap_points(y_inset)
    else:
        pt_start, pt_end = None, None

    return solid_from_ribs(ribs, pt_start, pt_end)


def _get_cap_points(ts: List[Tuple]) -> Tuple[Point3, Point3]:
    """
    Get cap points from a solid descriptor.

    Uses solid descriptor for y values and x=0, z=0

    """
    pt_start = vec3(0, ts[0][0], 0)
    pt_end = vec3(0, ts[-1][0], 0)
    return pt_start, pt_end


def link(idxs_0: List[int], idxs_1: List[int]) -> Faces:
    """generate faces that link two corresponding sets of vertices"""

    assert len(idxs_0) == len(idxs_1)

    idxs = list(range(len(idxs_0)))
    faces_list = []
    for idx_a, idx_b in zip(idxs[:-1], idxs[1:]):

        # for now we want to assume that the new faces we are making are convex
        # so adding some extra logic
        # we could always make a version that doesn't do this

        faces_list.append(
            verts_quad([idxs_0[idx_a], idxs_0[idx_b], idxs_1[idx_b], idxs_1[idx_a]]))

    return np.concatenate(faces_list, axis=0)


def link_map(idxs_0: List[int], idxs_1: List[int], mapping: List[Tuple]) -> Faces:
    """generate faces that link two corresponding sets of vertices"""

    faces_list = []

    for idx in range(len(mapping)):
        m_0, m_1 = mapping[idx]

        if isinstance(m_0, int):
            one, many = m_0, m_1
            v_i_a, v_i_b = idxs_0, idxs_1
        elif isinstance(m_1, int):
            many, one = m_0, m_1
            v_i_b, v_i_a = idxs_0, idxs_1

        # TODO: the case where we have two single indices later
        # TODO: this might be backward

        # create a triangle for every pair of indices in many
        for idx_a, idx_b in zip(many[:-1], many[1:]):
            faces_list.append(np.array([v_i_a[one], v_i_b[idx_b], v_i_b[idx_a]]))

            # these might be used to automatically link previous faces
            r_a = v_i_a[one]
            r_b = v_i_b[idx_a]

    return np.stack(faces_list, axis=0)

# ~~~~ functions for creating meshes from extending sequences of loops


def cat(org: np.ndarray, new: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """concatenate a new array onto an existing array, and also return
    the indices of the new array"""
    verts = np.concatenate([org, new], axis=0)
    return verts, list(range(org.shape[0], org.shape[0] + new.shape[0]))


def extend_link(
        mesh: Mesh,
        vert_idxs_old: List[int],
        verts_new: np.ndarray) -> MeshExtended:
    """extend a mesh, kind of like extrude"""

    return extend(
        mesh,
        vert_idxs_old,
        verts_new,
        link)


def extend_link_loop(
        mesh: Mesh,
        vert_idxs_old: List[int],
        verts_new: np.ndarray) -> MeshExtended:
    """extend a mesh, assuming the vertices we are extending / from to
    should be treated as loops"""

    return extend(
        mesh,
        vert_idxs_old,
        verts_new,
        link_loop)


def link_loop(x: List[int], y: List[int]) -> Faces:
    """loop each list of vertices before linking"""
    return link(util.loop(x), util.loop(y))


def extend_to_cap(
        mesh: Mesh,
        vert_idxs_old: List[int],
        vert_new: Point3) -> MeshExtended:
    """extend a mesh, capping off an open loop"""

    return extend(
        mesh,
        vert_idxs_old,
        np.array([vert_new]),
        lambda x, y: close_face_noloop(x, y[0]))


def extend_from_cap(
        mesh: Mesh,
        vert_idx_old: int,
        verts_new: np.ndarray,
        do_loop: bool) -> MeshExtended:
    """extend a mesh from a single point"""

    def call(x, y):
        if do_loop:
            # TODO: this here as well
            y_mod = util.loop(y)
        else:
            y_mod = y
        return close_face_noloop(list(reversed(y_mod)), x[0])

    return extend(
        mesh,
        [vert_idx_old],
        verts_new,
        call)


def extend(
        mesh: Mesh,
        vert_idxs_old: List[int],
        verts_new: Verts,
        extend_func: Callable) -> MeshExtended:
    """generic mesh update function"""

    verts, faces = mesh

    verts, vert_idxs_new = cat(verts, verts_new)

    faces_new = extend_func(vert_idxs_old, vert_idxs_new)
    faces, face_idxs_new = cat(faces, faces_new)

    return (verts, faces), vert_idxs_new, face_idxs_new


def extend_existing(
        mesh: Mesh,
        vert_idxs_old_0: List[int],
        vert_idxs_old_1: List[int],
        extend_func) -> Tuple[Mesh, List[int]]:
    """Connect existing points to existing points."""

    verts, faces = mesh
    faces_new = extend_func(vert_idxs_old_0, vert_idxs_old_1)
    faces, face_idxs_new = cat(faces, faces_new)

    return (verts, faces), face_idxs_new


def verts_quad(idxs: Union[List[int], np.ndarray]) -> Faces:
    """make a quad (two triangles) from indices"""

    return np.vstack((
        idxs[0:3],
        (idxs[2], idxs[3], idxs[0])))


def close_face_loop(idxs_points: List[int], idx_center: int) -> Faces:
    """generate triangles to link an ordered set of points to a center point"""

    idxs_vert = list(idxs_points)
    idxs_vert.append(idxs_vert[0])

    faces_list = []
    for idx_vert_a, idx_vert_b in zip(idxs_vert[:-1], idxs_vert[1:]):
        faces_list.append((idx_center, idx_vert_a, idx_vert_b))

    return np.vstack(faces_list)


def close_face_noloop(idxs_points: List[int], idx_center: int) -> Faces:
    """generate triangles to link an ordered set of points to a center point"""

    idxs_vert = list(idxs_points)

    faces_list = []
    for idx_vert_a, idx_vert_b in zip(idxs_vert[:-1], idxs_vert[1:]):
        faces_list.append((idx_center, idx_vert_a, idx_vert_b))

    return np.vstack(faces_list)


def surface_revolution(
        xy_points: np.ndarray,
        angle: float,
        close_rev: bool,
        close_fn: bool,
        subdivisions: int) -> Mesh:

    """create a surface of revolution by rotating xy points around the y axis"""

    # TODO: handle zero points properly without duplicating them

    # pylint: disable=too-many-locals

    # for now, only the y axis
    # y axis remains fixed, x is modified and z is added
    n_points = xy_points.shape[0]

    # add points for the final subdivision if isn't a full revolution
    max_sub = subdivisions + 1 if angle < (np.pi * 2.0) else subdivisions

    # for now, just lay them out in a line
    points_list = []

    for x_val in range(0, max_sub):
        theta = angle * x_val / subdivisions
        x_dim = xy_points[:, 0] * np.cos(theta)
        y_dim = xy_points[:, 1]
        z_dim = xy_points[:, 0] * np.sin(theta)

        points_list.append(
            np.column_stack((x_dim, y_dim, z_dim)))

    # connect corresponding pairs of points with a face
    faces_list = []

    idxs_sub = list(range(0, max_sub))
    if close_rev:
        idxs_sub.append(0)
    idxs_vert = list(range(0, n_points))
    if close_fn:
        idxs_vert.append(0)

    for idx_sub_a, idx_sub_b in zip(idxs_sub[:-1], idxs_sub[1:]):
        for idx_vert_a, idx_vert_b in zip(idxs_vert[:-1], idxs_vert[1:]):
            offset_a = idx_sub_a * n_points
            offset_b = idx_sub_b * n_points
            faces_list.append(
                [offset_a + idx_vert_a, offset_a + idx_vert_b, offset_b + idx_vert_a])
            faces_list.append(
                [offset_b + idx_vert_a, offset_a + idx_vert_b, offset_b + idx_vert_b])

    # TODO: some degenerate faces are produced here I think

    return np.vstack(points_list), np.vstack(faces_list)


def solid_from_cuts(
        dims: Vec3,
        planes: List[Tuple[Point3, Vec3]]
        ) -> Tuple[Mesh, List[Mesh]]:
    """create a solid from cuboid and a bunch of cutting planes"""

    from modeling import primitives
    import trimesh.geometry

    cutter_scale = 2.5
    mesh = primitives.box_mesh(*dims)

    max_dim = max(dims) * cutter_scale
    cutter = primitives.box_mesh(max_dim, max_dim, max_dim)
    cutter = ops.translate_mesh(cutter, vec3(0, 0, max_dim / 2))

    cutter_vis = ops.scale_mesh(cutter, vec3(1, 1, 0.01))

    cutters = []

    for pos, normal in planes:
        normal = normal / np.linalg.norm(normal)
        rot = trimesh.geometry.align_vectors(vec3(0, 0, 1), normal)[0:3, 0:3]

        # do the cutting
        cutter_transf = ops.rotate_mesh(cutter, rot)
        cutter_transf = ops.translate_mesh(cutter_transf, pos)
        mesh = util.difference([mesh, cutter_transf])

        # add a scaled version of the cutter for viewing purposes
        cutter_transf = ops.rotate_mesh(cutter_vis, rot)
        cutter_transf = ops.translate_mesh(cutter_transf, pos)
        cutters.append(cutter_transf)

    return mesh, cutters


def solid_from_polygon_triangular(
        pts: np.ndarray,
        width: float,
        height: float
        ) -> Mesh:
    """create a solid from a 2D polygon"""

    assert pts.shape[1] == 2

    # TODO: do without repeated rib / process
    cutter = solid_from_ribs(
        [
            util.attach_z(pts, -height / 2),
            util.attach_z(util.inset_polygon(pts, width / 2), height / 2),
            util.attach_z(util.inset_polygon(pts, -width / 2), height / 2),
            util.attach_z(pts, -height / 2)  # hack hack hack
        ],
        None,
        None
    )
    cutter = util.process(cutter)

    return cutter


def solid_from_polygon_rectangular(
        pts: np.ndarray,
        width: float,
        height: float
        ) -> Mesh:
    """create a solid from a 2D polygon"""

    assert pts.shape[1] == 2

    inset_pos = util.inset_polygon(pts, width / 2)
    inset_neg = util.inset_polygon(pts, -width / 2)

    # TODO: do without repeated rib / process
    cutter = solid_from_ribs(
        [
            util.attach_z(inset_pos, -height / 2),
            util.attach_z(inset_pos, height / 2),
            util.attach_z(inset_neg, height / 2),
            util.attach_z(inset_neg, -height / 2),
            util.attach_z(inset_pos, -height / 2)  # hack hack hack
        ],
        None,
        None
    )
    cutter = util.process(cutter)

    return cutter
