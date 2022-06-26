"""

Common functions for nicer dealings with mesh objects.

"""

# Copyright (c) 2019 Ben Zimmer. All rights reserved.

import math
from typing import Any, Union, Tuple, List, Callable, Optional

import trimesh
import trimesh.transformations
import trimesh.intersections
import trimesh.boolean

from matplotlib import pyplot as plt
import numpy as np
import pyrender

from modeling import view, ops
from modeling.types import Mesh, Verts, Verts2D, Point3, AABB, Vec3


def tm(verts, faces):  # pylint: disable=invalid-name
    """trimesh constructor"""
    return trimesh.Trimesh(verts, faces, process=False)


def concat(meshes: List[Mesh]) -> Mesh:
    """concatenate meshes"""
    # TODO: do concat myself without the conversion to trimesh and back
    tm_mesh = concat_tm([trimesh.Trimesh(vertices=x[0], faces=x[1], process=False) for x in meshes])
    # TODO: add assertion that the total counts of vertices and faces are correct
    return tm_mesh.vertices, tm_mesh.faces


def concat_tm(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    """concatenate trimeshes"""
    return trimesh.util.concatenate(meshes)


def loop(xs: Any) -> List:
    """append the starting element of a list to the end"""
    xs = list(xs)
    return xs + [xs[0]]


def rev(xs: Any) -> List:
    """convert to a list and reverse"""
    xs = list(xs)
    return list(reversed(xs))


def spin(mesh, n_dups, offset):
    """repeate meshes in a circle"""
    # also sometimes known as "circular array"
    return [
        ops.rotate_mesh(mesh, ops.rotation_y(x / n_dups * ops.FULL + offset))
        for x in range(n_dups)]


def spin_frac(mesh, n_dups, offset, frac):
    """repeat meshes in a fraction of a circle, including start and end"""
    return [
        ops.rotate_mesh(mesh, ops.rotation_y(x / (n_dups - 1) * frac + offset))
        for x in range(n_dups)]


def attach_x(yz: np.ndarray, x: float) -> np.ndarray:
    """add a z dimension to a set of 2D points"""
    xs = np.ones(yz.shape[0]) * x
    xs = np.expand_dims(xs, axis=1)
    return np.concatenate([xs, yz], axis=1)


def attach_z(xy: np.ndarray, z: float) -> np.ndarray:
    """add a z dimension to a set of 2D points"""
    zs = np.ones(xy.shape[0]) * z
    zs = np.expand_dims(zs, axis=1)
    return np.concatenate([xy, zs], axis=1)


def attach_y(xz: np.ndarray, y: float) -> np.ndarray:
    """add a y dimension to a set of 2D points"""
    # TODO: generalize and refactor

    ys = np.ones(xz.shape[0]) * y
    ys = np.expand_dims(ys, axis=1)

    return np.concatenate([xz[:, 0:1], ys, xz[:, 1:2]], axis=1)


# ~~~~ operations on vertices

def shift(xyz: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """translate all dimensions"""
    return xyz + trans


def elongate(
        pts: np.ndarray,
        amount: Union[Tuple, np.ndarray]) -> np.ndarray:
    """add sign-aware offsets"""
    amount = np.expand_dims(amount, axis=0)
    return pts + np.sign(pts) * amount

# ~~~~ 2D primitives ~~~~ ~~~~ ~~~~


def circle_point(angle: float, radius: float) -> Tuple[float, float]:
    """get a point on a circle"""
    x_coord = radius * math.cos(angle)
    y_coord = radius * math.sin(angle)
    return x_coord, y_coord


def points_around_circle(n_points: int, start: float, radius: float) -> List[Tuple[float, float]]:
    """find a list of evenly space points around a circle"""
    return [
        circle_point(x * 1.0 * ops.TAU / n_points + start, radius)
        for x in range(n_points)]


def ngon(n: int, start: float) -> Verts2D:
    """le sigh"""
    return np.array(points_around_circle(n, start, 1))


def rect_2d(width, height) -> Verts2D:
    """rectangular rib in xy"""

    width_half = width * 0.5
    height_half = height * 0.5

    return np.array([
        (width_half, -height_half),
        (width_half, height_half),
        (-width_half, height_half),
        (-width_half, -height_half)
    ])


def rect_bevel_2d(width, height, bevel) -> Verts2D:
    """get a rib, parameterized by the height and width of the opening"""

    # TODO: there's a generic bevel somewhere in here

    width_half = width * 0.5
    height_half = height * 0.5

    return np.array([
        (width_half - bevel, -height_half),
        (width_half, -height_half + bevel),
        (width_half, height_half - bevel),
        (width_half - bevel, height_half),

        # TODO: function for this - flip sign and reverse order
        (-width_half + bevel, height_half),
        (-width_half, height_half - bevel),
        (-width_half, -height_half + bevel),
        (-width_half + bevel, -height_half)
    ])

# ~~~~ ~~~~ ~~~~ ~~~~


def reverse_mapping(xs):
    xs_rev = []
    for x, y in xs:
        if isinstance(x, tuple):
            x = tuple(reversed(x))
        if isinstance(y, tuple):
            y = tuple(reversed(y))
        xs_rev.append((y, x))
    return xs_rev


def face_normal(tri_verts):
    """find the normal direction of a face"""
    v0 = tri_verts[0, :]
    v1 = tri_verts[1, :]
    v2 = tri_verts[2, :]
    return np.cross(v1 - v0, v2 - v0)


def union(meshes: List[Mesh]) -> Mesh:
    """union using scad"""
    meshes_tm = [tm(*x) for x in meshes]
    res = trimesh.boolean.union(meshes_tm, engine='scad', debug=False)
    return res.vertices, res.faces


def difference(meshes: List[Mesh]):
    """difference using scad"""
    meshes_tm = [tm(*x) for x in meshes]
    res = trimesh.boolean.difference(meshes_tm, engine='scad', debug=False)
    return res.vertices, res.faces


def face_colors(color, length):
    """face colors"""
    return np.repeat([color], length, axis=0)


# TODO: this ffmpeg function should go somewhere else

def ffmpeg_command(images_dirname, output_filename, width, height, fps):
    """prepare a command for ffmpeg"""
    command = (
        "ffmpeg -y -r " + str(fps) +
        " -f image2 -s " + str(width) + "x" + str(height) +
        " -i " + images_dirname + "/%04d.png " +
        "-threads 2 -vcodec libx264 -crf 25 -pix_fmt yuv420p " + output_filename)
    return command


def view_mesh(mesh: Any) -> None:
    """quick and dirty view"""
    material = pyrender.material.MetallicRoughnessMaterial(
        baseColorFactor=(180, 180, 190),
        metallicFactor=0.25,
        roughnessFactor=0.5)

    print('interactive viewer')
    print('\ti - cycle axis options')
    print('\tw - wireframe')
    print('\tq - quit')

    view.view(mesh, material)


def plot_loop_2d(loop_verts):
    """plot a 2d loop for debugging"""
    plt.figure()
    plt.axis('equal')
    plt.fill(
        loop_verts[:, 0], loop_verts[:, 1],
        facecolor='none', edgecolor='blue', linewidth=2)
    plt.scatter(
        loop_verts[:, 0], loop_verts[:, 1],
        color='blue')
    plt.show()


def plot_points_2d(points):
    """plot a 2d loop for debugging"""
    plt.figure()
    plt.axis('equal')
    plt.scatter(
        points[:, 0], points[:, 1],
        color='blue')
    plt.show()


def adjust_verts(verts: Verts, select: Callable, modify: Callable) -> Verts:
    """adjust verts by selecting and modifying vertices"""
    verts = np.array(verts)
    vert_idxs = select(verts)
    verts_selected = verts[vert_idxs, :]
    verts_selected_modified = modify(verts_selected)
    verts[vert_idxs, :] = verts_selected_modified
    return verts


def adjust_mesh(mesh: Mesh, select: Callable, modify: Callable) -> Mesh:
    """for convenience"""
    verts, faces = mesh
    verts = adjust_verts(verts, select, modify)
    return verts, faces


# stuff for inset

def inset_polygon(pts, amount) -> Verts:
    """find an inset version of an arbitrary polygon"""
    pts_a = [pts[-1]] + list(pts[:-1])
    pts_b = list(pts)
    pts_c = list(pts[1:]) + [pts[0]]
    return np.array([
        inset_point(a, b, c, amount)
        for a, b, c in zip(pts_a, pts_b, pts_c)])


def inset_quad(pt_a, pt_b, pt_c, pt_d, amount) -> Verts:
    """find an insert version of a quad"""
    # the points go like this:
    # b  a
    # c  d

    angles = [
        (pt_d, pt_a, pt_b),
        (pt_a, pt_b, pt_c),
        (pt_b, pt_c, pt_d),
        (pt_c, pt_d, pt_a)
    ]

    inset_pts = np.array([inset_point(*angle, amount) for angle in angles])

    return inset_pts


def inset_point(pt_a, pt_b, pt_c, amount):
    """given three points in counterclockwise order,
    find an inset version of the second point"""

    #     b
    #     .
    # c       a

    vec_0 = pt_a - pt_b
    vec_0 = vec_0 / np.linalg.norm(vec_0)
    vec_1 = pt_c - pt_b
    vec_1 = vec_1 / np.linalg.norm(vec_1)

    # there's probably a way to do this without trig / inverse
    # but I won't worry about that for now
    vec_dot = np.dot(vec_0, vec_1)

    theta = math.acos(vec_dot)
    half_theta = 0.5 * theta

    offset = amount / math.tan(half_theta)
    offset_vec = pt_b + vec_0 * offset

    # # extra = np.array([
    # #     pt_b + vec_0,
    # #     pt_b + vec_1,
    # #     pt_b + perp_vec
    # #
    # # ])
    # plt.figure()
    # plt.axis('equal')
    # abc = np.array([pt_a, pt_b, pt_c])
    # plt.scatter(abc[:, 0], abc[:, 1], color='blue')
    # # plt.scatter(res[0], res[1], color='green')
    # # plt.scatter(extra[:, 0], extra[:, 1], color='red')
    # plt.show()

    # now just have to add the amount perpendicular to the offset
    perp_vec = vec_1 - vec_0 * vec_dot
    perp_vec = perp_vec / np.linalg.norm(perp_vec)

    res = offset_vec + perp_vec * amount

    return res


def bevel_polygon(pts: Verts, bevel_amount: float, inset_amount: Optional[float]) -> Verts:
    """bevel a polygon"""

    pts_a = [pts[-1]] + list(pts[:-1])
    pts_b = list(pts)
    pts_c = list(pts[1:]) + [pts[0]]

    pts_bev = [
        bevel_point(a, b, c, bevel_amount)
        for a, b, c in zip(pts_a, pts_b, pts_c)
    ]

    if inset_amount is not None:
        pts_inset = [
            inset_point(a, b, c, inset_amount)
            for a, b, c in zip(pts_a, pts_b, pts_c)
        ]
        groups = [
            (a, b, c)
            for (a, c), b in zip(pts_bev, pts_inset)
        ]
    else:
        groups = pts_bev

    return np.array([y for x in groups for y in x])


def bevel_point(pt_a, pt_b, pt_c, amount):
    """

    Given 3 points, find two additional points on either side of the central point (b)

    """

    #     b
    #     .
    # c       a

    pt_a_bev_diff = pt_a - pt_b
    pt_a_bev_diff = pt_a_bev_diff / np.linalg.norm(pt_a_bev_diff)
    pt_a_bev = pt_b + pt_a_bev_diff * amount

    pt_c_bev_diff = pt_c - pt_b
    pt_c_bev_diff = pt_c_bev_diff / np.linalg.norm(pt_c_bev_diff)
    pt_c_bev = pt_b + pt_c_bev_diff * amount

    return pt_a_bev, pt_c_bev


def symmetrize(mesh: Mesh, plane_normal) -> Mesh:
    """symmetrize mesh around origin in any direction"""

    tm_mesh = tm(*mesh)
    tm_mesh_half = trimesh.intersections.slice_mesh_plane(
        tm_mesh,
        plane_normal,
        (0, 0, 0)
    )

    # verts_flipped = tm_mesh_half.vertices * [[-1, 1, 1]]
    verts_flipped = tm_mesh_half.vertices - 2.0 * (
            np.expand_dims(np.dot(tm_mesh_half.vertices, plane_normal), axis=1) * [plane_normal])
    tm_mesh_half_flipped = trimesh.Trimesh(
        vertices=verts_flipped,
        faces=np.flip(tm_mesh_half.faces, axis=1),
        process=False
    )

    tm_mesh_whole = concat_tm([tm_mesh_half, tm_mesh_half_flipped])
    print('before process:', tm_mesh_whole.vertices.shape, tm_mesh_whole.faces.shape)
    tm_mesh_whole = trimesh.Trimesh(
        tm_mesh_whole.vertices,
        tm_mesh_whole.faces,
        process=True)
    print('after process:', tm_mesh_whole.vertices.shape, tm_mesh_whole.faces.shape)

    return tm_mesh_whole.vertices, tm_mesh_whole.faces


def remove_bad_faces(faces: np.ndarray) -> np.ndarray:
    """remove degenerate faces"""
    keep = []
    for face in faces:
        if len(set(face)) < 3:
            keep.append(False)
        else:
            keep.append(True)

    print(np.sum(keep), '/', faces.shape[0], 'good')

    return faces[keep, :]


def process(mesh: Mesh) -> Mesh:
    """use trimesh's process"""
    # NOTE: validate=True added 2022-03-15
    mesh_tm = trimesh.Trimesh(mesh[0], mesh[1], process=True, validate=True)
    return mesh_tm.vertices, mesh_tm.faces


def point_mesh(point: Point3) -> Mesh:
    """create a mesh from a single point"""
    point = np.array(point, dtype=np.float)
    verts = np.array([point])
    faces = np.zeros((0, 3), dtype=np.int)
    mesh = (verts, faces)
    return mesh


def verts_bounds(verts: Verts) -> AABB:
    """find verts AABB"""
    return np.concatenate([
        np.min(verts, axis=0, keepdims=True),
        np.max(verts, axis=0, keepdims=True)], axis=0)


def mesh_bounds(mesh: Mesh) -> AABB:
    """find mesh AABB, for convenience"""
    return verts_bounds(mesh[0])


def bounds_overlap(bounds_0: AABB, bounds_1: AABB) -> bool:
    """check overlap of two AABBs"""
    return np.all(np.logical_and(bounds_0[0, :] < bounds_1[1, :], bounds_0[1, :] > bounds_1[0, :]))


def autosmooth(mesh: Mesh, angle: float) -> Mesh:
    """a wrapper around trimeshes autosmooth-like functionality"""
    smoothed_tm = tm(*mesh).smoothed(angle=angle)
    return smoothed_tm.vertices, smoothed_tm.faces


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