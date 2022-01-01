"""
Read PLY files.
"""

# Copyright (c) 2022 Ben Zimmer. All rights reserved.

from typing import Tuple, Optional

import numpy as np
import plyfile

from modeling.types import Mesh


def read(file_path: str) -> Tuple[Mesh, Optional[np.ndarray]]:
    """read a ply file"""
    data = plyfile.PlyData.read(file_path)

    vertices = np.stack([
        data['vertex']['x'],
        data['vertex']['y'],
        data['vertex']['z'],
    ], axis=1)

    if 'red' in data['vertex']:
        colors = np.stack([
            data['vertex']['red'],
            data['vertex']['green'],
            data['vertex']['blue'],
            data['vertex']['alpha']
        ], axis=1)
    else:
        colors = None

    faces = np.vstack([x[0] for x in data['face']])

    return (vertices, faces), colors
