"""

Tests for blender utilities.

"""

# Copryight (c) 2020 Ben Zimmer. All rights reserved.

from unittest.mock import Mock

from modeling import blender


def test_flatten():
    """test flattening node structures"""

    n_a = 'a'
    n_b = 'b'
    n_c = 'c'

    struct = [n_a, n_b, n_c]
    assert blender.flatten_nodes(struct) == struct

    struct = [[n_a, n_b, n_c], n_b, n_c]
    res = [n_a, n_b, n_c, n_b, n_c]
    assert blender.flatten_nodes(struct) == res

    struct = [[[n_a, n_b], n_a, n_b], n_a, n_b]
    res = [n_a, n_b, n_a, n_b, n_a, n_b]
    assert blender.flatten_nodes(struct) == res


def test_configure_cycles():
    """test configure_cycles"""

    scene = Mock()

    blender.configure_cycles(scene, 16, 32, True)

    # not the best test but whatever
    assert scene.render.engine == 'CYCLES'
    assert scene.cycles.feature_set == 'EXPERIMENTAL'
    assert scene.cycles.samples == 16
    assert scene.cycles.preview_samples == 32
    assert scene.cycles.device == 'GPU'
