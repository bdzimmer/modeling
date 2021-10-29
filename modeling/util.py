"""

Common functions for nicer dealings with mesh objects.

"""

# Copyright (c) 2019 Ben Zimmer. All rights reserved.

import math
from typing import Any, Union, Tuple, List, Callable

import trimesh
import trimesh.transformations
import trimesh.intersections
from matplotlib import pyplot as plt
import numpy as np
import pyrender

from modeling import view
from modeling.types import Mesh, Vec3, Verts

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])

X_HAT_NEG = np.array([-1.0, 1.0, 1.0])
Z_HAT_NEG = np.array([1.0, 1.0, -1.0])

FULL = 2.0 * np.pi
TAU = FULL


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


def rotation_x(angle):
    """create matrix for rotation around x axis"""
    return trimesh.transformations.rotation_matrix(angle, X_HAT)


def rotation_y(angle):
    """create matrix for rotation around y axis"""
    return trimesh.transformations.rotation_matrix(angle, Y_HAT)


def rotation_z(angle):
    """create matrix for rotation around z axis"""
    return trimesh.transformations.rotation_matrix(angle, Z_HAT)


# TODO: still not clear on whether the copy() calls are required

def translate(
        mesh: Mesh,
        trans: Vec3) -> Mesh:
    """translate a mesh"""
    return mesh[0] + trans, mesh[1]


def rotate(mesh: Mesh, rot_mat: np.ndarray) -> Mesh:
    """rotate verts"""
    verts, faces = mesh
    verts = np.transpose(np.dot(rot_mat, np.transpose(verts)))
    return verts, faces


def rotate_verts(verts: Verts, rot_mat: np.ndarray) -> np.ndarray:
    """rotate verts"""
    return np.transpose(np.dot(rot_mat, np.transpose(verts)))


def transform(mesh: Mesh, mat: np.ndarray) -> Mesh:
    """transform mesh"""
    verts, faces = mesh

    rot = mat[0:3, 0:3]
    trans = mat[0:3, 3:4]

    verts_new = np.transpose(
        np.dot(rot, np.transpose(verts)) + trans)

    return verts_new, faces

    # gut check
    # trimesh = tm(*mesh)
    # trimesh = trimesh.copy().apply_transform(mat)
    # return trimesh.vertices, trimesh.faces


def spin(mesh, n_dups, offset):
    """repeate meshes in a circle"""
    # also sometimes known as "circular array"
    return [
        transform(mesh, rotation_y(x / n_dups * FULL + offset))
        for x in range(n_dups)]


def spin_frac(mesh, n_dups, offset, frac):
    """repeat meshes in a fraction of a circle, including start and end"""
    return [
        transform(mesh, rotation_y(x / (n_dups - 1) * frac + offset))
        for x in range(n_dups)]


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

# TODO: update signatures

def scale_xz(xyz: np.ndarray, scale: float) -> np.ndarray:
    """scale the xz dimensions of 3D points"""
    xz = xyz[:, [0, 2]]
    xz = xz * scale
    return np.concatenate([xz[:, 0:1], xyz[:, 1:2], xz[:, 1:2]], axis=1)


def scale(xyz: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """scale all dimensions"""
    return xyz * scale


def shift(xyz: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """translate all dimensions"""
    return xyz + trans


def elongate(
        pts: np.ndarray,
        amount: Union[Tuple, np.ndarray]) -> np.ndarray:
    """add sign-aware offsets"""
    amount = np.expand_dims(amount, axis=0)
    return pts + np.sign(pts) * amount

# ~~~~ ~~~~ ~~~~ ~~~~


def rect_2d(width, height):
    """rectangular rib in xy"""

    width_half = width * 0.5
    height_half = height * 0.5

    return np.array([
        (width_half, -height_half),
        (width_half, height_half),
        (-width_half, height_half),
        (-width_half, -height_half)
    ])


def rect_bevel_2d(width, height, bevel):
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
    meshes_tm = [trimesh.Trimesh(*x) for x in meshes]
    res = trimesh.boolean.union(meshes_tm, engine='scad')
    return res.vertices, res.faces


def difference(meshes: List[Mesh]):
    """difference using scad"""
    meshes_tm = [trimesh.Trimesh(*x) for x in meshes]
    res = trimesh.boolean.difference(meshes_tm, engine='scad')
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


def mirror_x(mesh: Mesh) -> Mesh:
    """mirror mesh in x dimension (across y axis)"""
    return (
        mesh[0] * np.array([X_HAT_NEG]),
        mesh[1][:, [2, 1, 0]]
    )


def mirror_z(mesh: Mesh) -> Mesh:
    """mirror mesh in z dimension (across x axis)"""
    return (
        mesh[0] * np.array([Z_HAT_NEG]),
        mesh[1][:, [2, 1, 0]]
    )


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


def bevel_polygon(pts, bevel_amount, inset_amount):
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
