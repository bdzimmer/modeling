"""

Types for meshes and related data.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from typing import Tuple, List, Union

import numpy as np

Point3 = Union[np.ndarray, Tuple[float, float, float]]
Vec3 = Union[np.ndarray, Tuple[float, float, float]]

Verts2D = np.ndarray
Verts = np.ndarray
Faces = np.ndarray

FaceColors = np.ndarray  # (n x 3) or (n x 4)

# mesh consists of faces and vertices
Mesh = Tuple[Verts, Faces]

# a new mesh with indices of new vertices and new faces
MeshExtended = Tuple[Mesh, List[int], List[int]]


# some shorthand for numpy array creation

def vec(*xs) -> np.ndarray:
    """shorthand for numpy array creation"""
    return np.array(xs)


def vec3(x, y, z) -> Vec3:
    """shorthand for a specific sized numpy array"""
    return np.array([x, y, z])
