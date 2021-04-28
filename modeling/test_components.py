"""

Tests for components.

"""

# Copyright (c) 2019 Ben Zimmer. All rights reserved.

import numpy as np
import trimesh

from modeling import components, view

FULL = 2.0 * np.pi

DEBUG_VISUALIZE = False


def test_curved_plate():
    """test for curved_plate"""

    subdivisions = 16

    # compound radius curved plate

    mesh = components.compound_radius_plate(
        10, 5, 10, 2,
        FULL / 5.0, 1, 1, subdivisions)

    mesh = trimesh.Trimesh(*mesh)

    if DEBUG_VISUALIZE:
        view.view(mesh, None)

    assert mesh.vertices.shape[0] == subdivisions * 4 + 4

    # single radius curved plate

    mesh = components.curved_plate(
        10, 5, 4, 5, 4,
        FULL / 5.0, 1, 1, subdivisions)

    mesh = trimesh.Trimesh(*mesh)

    if DEBUG_VISUALIZE:
        view.view(mesh, None)

    assert mesh.vertices.shape[0] == subdivisions * 4 + 4
