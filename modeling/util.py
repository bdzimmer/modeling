"""

Common functions for nicer dealings with mesh objects.

"""

# Copyright (c) 2019 Ben Zimmer. All rights reserved.

from typing import Any, Union, Tuple, List

import trimesh
from trimesh import transformations
import numpy as np
import pyrender

from modeling import view
from modeling.types import Mesh, Vec3

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])

X_HAT_NEG = np.array([-1.0, 1.0, 1.0])


FULL = 2.0 * np.pi
TAU = FULL


def tm(verts, faces):  # pylint: disable=invalid-name
    """trimesh constructor"""
    return trimesh.Trimesh(verts, faces, process=False)


def concat(meshes: List[Mesh]) -> Mesh:
    """concatenate meshes"""
    # TODO: do this myself without the conversion to trimesh and back
    tm_mesh = concat_tm([trimesh.Trimesh(vertices=x[0], faces=x[1], process=False) for x in meshes])
    # TODO: add assertion that the total counts of vertices and faces are correct
    return tm_mesh.vertices, tm_mesh.faces


def concat_tm(meshes):
    """for now"""
    return trimesh.util.concatenate(meshes)


def loop(xs: Any) -> List:
    xs = list(xs)
    return xs + [xs[0]]


def rev(xs: Any) -> List:
    xs = list(xs)
    return list(reversed(xs))


def rotation_x(angle):
    """create matrix for rotation around x axis"""
    return transformations.rotation_matrix(angle, X_HAT)


def rotation_y(angle):
    """create matrix for rotation around y axis"""
    return transformations.rotation_matrix(angle, Y_HAT)


def rotation_z(angle):
    """create matrix for rotation around z axis"""
    return transformations.rotation_matrix(angle, Z_HAT)


# TODO: still not clear on whether the copy() calls are required

def translate(
        mesh: Mesh,
        trans: Vec3) -> Mesh:
    """translate a mesh"""
    return mesh[0] + trans, mesh[1]


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


# also sometimes known as "circular array"

def spin(mesh, n_dups, offset):
    return [
        transform(mesh, rotation_y(x / n_dups * FULL + offset))
        for x in range(n_dups)]


# TODO: rename to "attach ___"

def add_z(xy: np.ndarray, z: float) -> np.ndarray:
    """add a z dimension to a set of 2D points"""
    zs = np.ones(xy.shape[0]) * z
    zs = np.expand_dims(zs, axis=1)
    return np.concatenate([xy, zs], axis=1)


def add_y(xz: np.ndarray, y: float) -> np.ndarray:
    """add a z dimension to a set of 2D points"""
    # TODO: generalize and refactor

    ys = np.ones(xz.shape[0]) * y
    ys = np.expand_dims(ys, axis=1)

    return np.concatenate([xz[:, 0:1], ys, xz[:, 1:2]], axis=1)


# ~~~~ operations on vertices

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
    """union with scad"""

    meshes_tm = []
    for mesh in meshes:
        meshes_tm.append(trimesh.Trimesh(*mesh))

    res = trimesh.boolean.union(meshes_tm, engine='scad')
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

    view.view(mesh, material)


def mirror_x(mesh: Mesh) -> Mesh:
    """mirror mesh in x dimension (across y axis)"""
    return (
        mesh[0] * np.array([X_HAT_NEG]),
        mesh[1][:, [2, 1, 0]]
    )
