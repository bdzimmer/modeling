"""
Tests for solid.

"""

# Copyright (c) 2022 Ben Zimmer. All rights reserved.

from modeling.types import vec3
from modeling import util, solid


def test_solid_from_cuts():
    """test solid_from_cuts"""

    mesh = solid.solid_from_cuts(
        vec3(2, 2, 2),
        [
            (vec3(0.5, 0.5, 0.5), vec3(1, 0.4, 1)),
            (vec3(0.5, 0.5, -0.5), vec3(1, 0.7, -1))
        ]
    )

    util.view_mesh(mesh)
