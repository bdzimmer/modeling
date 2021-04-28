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


# TODO: types for vert and face indices

# mesh consists of faces and vertices
Mesh = Tuple[Verts, Faces]

# a new mesh with indices of new vertices and new faces
MeshExtended = Tuple[Mesh, List[int], List[int]]
