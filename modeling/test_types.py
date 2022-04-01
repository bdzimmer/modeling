"""

Tests for types.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from modeling import types

import numpy as np


def test_types():
    """test types"""

    verts = np.zeros((10, 3), dtype=np.float)
    faces = np.zeros((20, 3), dtype=np.int)

    assert isinstance(verts, types.Verts)
    assert isinstance(faces, types.Faces)

    assert isinstance(types.vec3(1, 2, 3), np.ndarray)
    assert isinstance(types.vec(1, 2, 3, 4, 5), np.ndarray)
