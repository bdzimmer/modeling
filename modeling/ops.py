"""
Various transformations and mesh operations.
"""

# Copyright (c) 2022 Ben Zimmer. All rights reserved.

import numpy as np
import trimesh

from modeling.types import Mat33, Mesh, Vec3, Verts

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])
X_HAT_NEG = np.array([-1.0, 1.0, 1.0])
Y_HAT_NEG = np.array([1.0, -1.0, 1.0])
Z_HAT_NEG = np.array([1.0, 1.0, -1.0])

TAU = 2.0 * np.pi
FULL = TAU


def rotation_x(angle: float) -> Mat33:
    """create matrix for rotation around x axis"""
    return trimesh.transformations.rotation_matrix(angle, X_HAT)[0:3, 0:3]


def rotation_y(angle: float) -> Mat33:
    """create matrix for rotation around y axis"""
    return trimesh.transformations.rotation_matrix(angle, Y_HAT)[0:3, 0:3]


def rotation_z(angle: float) -> Mat33:
    """create matrix for rotation around z axis"""
    return trimesh.transformations.rotation_matrix(angle, Z_HAT)[0:3, 0:3]


ROTATION_X_90 = rotation_x(TAU / 4)
ROTATION_X_NEG_90 = rotation_x(-TAU / 4)

ROTATION_Y_90 = rotation_y(TAU / 4)
ROTATION_Y_NEG_90 = rotation_y(-TAU / 4)

ROTATION_Z_90 = rotation_z(TAU / 4)
ROTATION_Z_NEG_90 = rotation_z(-TAU / 4)

ROTATION_X_180 = rotation_x(TAU / 2)
ROTATION_Y_180 = rotation_y(TAU / 2)
ROTATION_Z_180 = rotation_z(TAU / 2)


def translate_mesh(mesh: Mesh, trans: Vec3) -> Mesh:
    """translate a mesh"""
    return mesh[0] + trans, mesh[1]


def rotate_mesh(mesh: Mesh, rot_mat: np.ndarray) -> Mesh:
    """rotate a mesh"""
    assert rot_mat.shape == (3, 3), 'rotate_mesh requires 3x3 matrix'

    verts, faces = mesh
    verts = rotate_verts(verts, rot_mat)
    return verts, faces


def scale_xz(xyz: np.ndarray, scale: float) -> np.ndarray:
    """scale the xz dimensions of 3D points"""
    xz = xyz[:, [0, 2]]
    xz = xz * scale
    return np.concatenate([xz[:, 0:1], xyz[:, 1:2], xz[:, 1:2]], axis=1)


def scale(xyz: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """scale all dimensions"""
    return xyz * scale


def scale_mesh(mesh: Mesh, amount: Vec3) -> Mesh:
    """scale a mesh"""
    return scale(mesh[0], amount), mesh[1]


def rotate_verts(verts: Verts, rot_mat: np.ndarray) -> np.ndarray:
    """rotate verts"""
    assert rot_mat.shape == (3, 3), 'rotate_verts requires 3x3 matrix'
    return np.transpose(np.dot(rot_mat, np.transpose(verts)))


def rotate_vec(vec: Vec3, rot_mat: np.ndarray) -> np.ndarray:
    """rotate a single vector"""
    assert rot_mat.shape == (3, 3), 'rotate_vec requires 3x3 matrix'
    return np.transpose(np.dot(rot_mat, np.transpose(np.array([vec]))))[0, :]


def transform_mesh(mesh: Mesh, mat: np.ndarray) -> Mesh:
    """transform mesh"""

    assert mat.shape == (4, 4), 'transform_mesh requires 4x4 matrix'

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


def mirror_x(mesh: Mesh) -> Mesh:
    """mirror mesh in x dimension (across y axis)"""
    return (
        mesh[0] * np.array([X_HAT_NEG]),
        mesh[1][:, [2, 1, 0]]
    )


def mirror_y(mesh: Mesh) -> Mesh:
    """mirror mesh in y dimension"""
    return (
        mesh[0] * np.array([Y_HAT_NEG]),
        mesh[1][:, [2, 1, 0]]
    )


def mirror_z(mesh: Mesh) -> Mesh:
    """mirror mesh in z dimension (across x axis)"""
    return (
        mesh[0] * np.array([Z_HAT_NEG]),
        mesh[1][:, [2, 1, 0]]
    )


def vapply(mesh: Mesh, fn, **kwargs) -> Mesh:
    """apply a function to vertices of a mesh"""
    return fn(mesh[0], **kwargs), mesh[1]


def fapply(mesh: Mesh, fn, **kwargs) -> Mesh:
    """apply a function to the faces of a mesh"""
    return mesh[0], fn(mesh[1], **kwargs)


def normalize(vec):
    """
    Normalize a vector.
    Doesn't check for zero divide.
    """
    return vec / np.linalg.norm(vec)
