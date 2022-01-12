"""

General components.

"""

# Copyright (c) 2019 Ben Zimmer. All rights reserved.

from typing import Tuple, List, Union, Callable

import numpy as np
import trimesh

from modeling import util
from modeling.types import Point3, Vec3, Verts2D, Verts, Faces, Mesh, MeshExtended


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


def close_face(idxs_points: List[int], idx_center: int) -> Faces:
    """generate triangles to link an ordered set of points to a center point"""

    # TODO: rewrite as "close_face_loop"

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


def verts_quad(idxs: Union[List[int], np.ndarray]) -> Faces:
    """make a quad (two triangles) from indices"""

    return np.vstack((
        idxs[0:3],
        (idxs[2], idxs[3], idxs[0])))


def tri_normal(verts: Union[List[Point3], Verts]) -> Vec3:
    """find the surface normal of a triangle"""
    seg_0 = verts[1] - verts[0]
    seg_1 = verts[2] - verts[0]
    res = np.cross(seg_0, seg_1)
    return res / np.linalg.norm(res)


def tri_area(verts: Union[List[Point3], Verts]) -> Vec3:
    """find the area of a triangle"""
    seg_0 = verts[1] - verts[0]
    seg_1 = verts[2] - verts[0]
    return 0.5 * np.linalg.norm(np.cross(seg_0, seg_1))


def is_triangle_pair_convex(
        a: Point3,
        b: Point3,
        c: Point3,
        d: Point3) -> bool:
    """test whether a pair of triangles (abc, adb) are convex or concave."""
    norm_0 = tri_normal([a, b, c])
    norm_1 = tri_normal([a, d, b])
    rot = np.cross(norm_0, norm_1)
    shared_edge = b - a
    return np.dot(rot, shared_edge) > 0


def quad_verts(quad_idxs: Faces) -> List[int]:
    """convert a quad back to vert indcices"""
    return [quad_idxs[0, 0], quad_idxs[0, 1], quad_idxs[0, 2], quad_idxs[1, 1]]


def cylinder(length: float, radius: float, subdivisions: int) -> Mesh:
    """cylinder with closed caps"""

    half_length = length / 2.0

    outline = np.array([
        [radius, -half_length],
        [radius, half_length]
    ])

    curve_verts, curve_faces = surface_revolution(
        outline, np.pi * 2.0, True, False, subdivisions)

    center_0 = np.array([[0.0, -half_length, 0.0]])
    center_1 = np.array([[0.0, half_length, 0.0]])

    center_0_idx = curve_verts.shape[0]
    center_1_idx = curve_verts.shape[0] + 1

    cap_0_idxs = range(0, subdivisions * 2, 2)
    cap_1_idxs = range(1, subdivisions * 2, 2)

    cap_0_faces = close_face(cap_0_idxs, center_0_idx)
    cap_1_faces = close_face(cap_1_idxs[::-1], center_1_idx)

    return (
        np.concatenate((curve_verts, center_0, center_1), axis=0),
        np.concatenate((curve_faces, cap_0_faces, cap_1_faces), axis=0))


def curved_plate(
        length: float,
        radius_outer_0: float,
        radius_inner_0: float,
        radius_outer_1: float,
        radius_inner_1: float,
        angle: float,
        chamfer_x: float,
        chamfer_y: float,
        subdivisions: int) -> Mesh:
    """create a curved plate"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # make a surface of revolution and cap the ends

    half_length = length / 2.0

    outline = np.array([
        [radius_inner_0, -half_length],
        [radius_outer_0, -half_length + chamfer_y],
        [radius_outer_1, half_length - chamfer_y],
        [radius_inner_1, half_length]
    ])

    verts, faces = surface_revolution(outline, angle, False, True, subdivisions)

    # TODO: chamfers could be done more efficiently
    if chamfer_x > 0:

        # TODO: it seems like this should still work with double-ended plates
        # angle_x = 2.0 * chamfer_x / radius_outer_0
        angle_x = 2.0 * np.arcsin(chamfer_x / 2.0 / radius_outer_0)
        print("chamfer angle 0:", angle_x * 180 / np.pi)
        verts_tightened, _ = surface_revolution(
            outline, angle - 2.0 * angle_x, False, True, subdivisions)
        verts_tightened = np.dot(verts_tightened, util.rotation_y(angle_x)[0:3, 0:3])

        idxs_0 = range(1, verts.shape[0], 4)
        verts[idxs_0, :] = verts_tightened[idxs_0, :]

        # angle_x = 2.0 * chamfer_x / radius_outer_1
        angle_x = 2.0 * np.arcsin(chamfer_x / 2.0 / radius_outer_1)
        print("chamfer angle 1:", angle_x * 180 / np.pi)
        verts_tightened, _ = surface_revolution(
            outline, angle - 2.0 * angle_x, False, True, subdivisions)
        verts_tightened = np.dot(verts_tightened, util.rotation_y(angle_x)[0:3, 0:3])

        idxs_1 = range(2, verts.shape[0], 4)
        verts[idxs_1, :] = verts_tightened[idxs_1, :]

    cap_0_idxs = np.array([3, 2, 1, 0])
    cap_1_idxs = subdivisions * 4 + np.array([0, 1, 2, 3])

    return verts, np.concatenate(
        (faces, verts_quad(cap_0_idxs), verts_quad(cap_1_idxs)))


def compound_radius_plate(
        length: float,
        radius_inner_0: float,
        radius_inner_1: float,
        thickness: float,
        angle: float,
        chamfer_x: float,
        chamfer_y: float,
        subdivisions: int) -> Mesh:
    """create a curved plate with a different radius at each end"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # make a surface of revolution and cap the ends

    half_length = length / 2.0

    compound_angle = np.arctan((radius_inner_1 - radius_inner_0) / length)
    chamfer_y_y = chamfer_y * np.cos(compound_angle)
    chamfer_y_x = chamfer_y * np.sin(compound_angle)
    thickness_x = thickness * np.cos(compound_angle)
    thickness_y = 0.0 - thickness * np.sin(compound_angle)

    print(thickness_x, thickness_y)

    outline = np.array([
        [radius_inner_0, -half_length],
        [
            radius_inner_0 + thickness_x + chamfer_y_x,
            -half_length + thickness_y + chamfer_y_y
        ],
        [
            radius_inner_1 + thickness_x - chamfer_y_x,
            half_length + thickness_y - chamfer_y_y
        ],
        [radius_inner_1, half_length]
    ])

    verts, faces = surface_revolution(
        outline, angle, False, True, # True
        subdivisions)

    # TODO: chamfers could be done more efficiently
    if chamfer_x > 0:

        # TODO: it seems like this should still work with double-ended plates
        # angle_x = 2.0 * chamfer_x / radius_outer_0
        angle_x = 2.0 * np.arcsin(chamfer_x / 2.0 / (radius_inner_0 + thickness))
        print("chamfer angle 0:", angle_x * 180 / np.pi)
        verts_tightened, _ = surface_revolution(
            outline, angle - 2.0 * angle_x, False, True, subdivisions)
        verts_tightened = np.dot(verts_tightened, util.rotation_y(angle_x)[0:3, 0:3])

        idxs_0 = range(1, verts.shape[0], 4)
        verts[idxs_0, :] = verts_tightened[idxs_0, :]

        # angle_x = 2.0 * chamfer_x / radius_outer_1
        angle_x = 2.0 * np.arcsin(chamfer_x / 2.0 / (radius_inner_1 + thickness))
        print("chamfer angle 1:", angle_x * 180 / np.pi)
        verts_tightened, _ = surface_revolution(
            outline, angle - 2.0 * angle_x, False, True, subdivisions)
        verts_tightened = np.dot(verts_tightened, util.rotation_y(angle_x)[0:3, 0:3])

        idxs_1 = range(2, verts.shape[0], 4)
        verts[idxs_1, :] = verts_tightened[idxs_1, :]

    cap_0_idxs = np.array([3, 2, 1, 0])
    cap_1_idxs = subdivisions * 4 + np.array([0, 1, 2, 3])

    return verts, np.concatenate(
        (faces, verts_quad(cap_0_idxs), verts_quad(cap_1_idxs)))


def circular_plating_mesh(
        plate_mesh_by_angle: Callable,
        angle_plate_total: float,
        angle_gap: float,
        num_plates: int) -> Mesh:
    """create a mesh for circular plating"""

    angle_plate = angle_plate_total / num_plates
    print(angle_plate * 180.0 / np.pi)
    plate = plate_mesh_by_angle(angle_plate - angle_gap)

    plates = []
    for x_val in range(num_plates):
        angle = x_val * angle_plate - 0.5 * angle_gap
        plates.append(util.rotate_mesh(plate, util.rotation_y(angle)))
    return util.concat(plates)


def box_mesh(x_extent: float, y_extent: float, z_extent: float) -> Mesh:
    """create a box mesh"""
    # wrapper around trimesh interface
    # TODO: my own implementation of this would be nice
    box = trimesh.primitives.Box(extents=(x_extent, y_extent, z_extent)).to_mesh()
    return box.vertices, box.faces


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


# functions for creating meshes from extending sequences of loops


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


# ~~~~ other


def solidify_tube(
        mesh: Mesh,
        subdivisions: int,
        scale_0: float,
        scale_1: float) -> Mesh:
    """make a surface of revolution into a solid object
    by scaling and connecting the ends"""

    verts, faces = mesh

    # number of verts in original thing before revolution
    n_verts = verts.shape[0] // subdivisions

    # find end indices
    end_idxs_0 = [x * n_verts for x in range(subdivisions)]
    end_idxs_1 = [x * n_verts + n_verts - 1 for x in range(subdivisions)]

    # scale each end and connect
    mesh, ex_idxs_0, _ = extend_link_loop(
        mesh, end_idxs_0, util.scale_xz(verts[end_idxs_0, :], scale_0))
    mesh, ex_idxs_1, _ = extend_link_loop(
        mesh, util.rev(end_idxs_1), util.scale_xz(verts[util.rev(end_idxs_1), :], scale_1))
    mesh, _ = extend_existing(mesh, ex_idxs_0, util.rev(ex_idxs_1), link_loop)

    return mesh


def solidify_loop(
        loop_xz: Verts2D,
        length: float) -> Mesh:
    half_length = 0.5 * length
    return solidify_loop_se(loop_xz, -half_length, half_length)


def solidify_loop_se(
        loop_xz: Verts2D,
        start_y: float,
        end_y: float) -> Mesh:
    """turn a 2d loop into a solid mesh"""

    point_x, point_z = np.mean(loop_xz, axis=0)

    point_start = np.array([point_x, start_y, point_z])
    point_end = np.array([point_x, end_y, point_z])

    mesh = point_mesh(point_start)

    mesh, ex_idxs, _ = extend_from_cap(mesh, 0, util.attach_y(loop_xz, start_y), True)
    mesh, ex_idxs, _ = extend_link_loop(mesh, ex_idxs, util.attach_y(loop_xz, end_y))
    mesh, ex_idxs, _ = extend_to_cap(mesh, util.loop(ex_idxs), point_end)

    return mesh


def point_mesh(point: Point3) -> Mesh:
    """create a mesh from a single point"""
    point = np.array(point, dtype=np.float)
    verts = np.array([point])
    faces = np.zeros((0, 3), dtype=np.int)
    mesh = (verts, faces)
    return mesh

# TODO: square plate with chamfered edges
# TODO: randomly generated plates and greebles
