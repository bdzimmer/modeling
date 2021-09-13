"""

Functions for loading materials into Blender.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.


import importlib
import sys
from typing import List, Tuple, Any, Union

import bpy.types as btypes

from modeling import blender


def _update_sys_path(path: str) -> None:
    """add modules to sys.path"""
    if path not in sys.path:
        print(f'adding {path} to sys.path')
        sys.path.append(path)


def material_python(
        name: str,
        obj: btypes.Object,
        func_desc: str,
        paths: List[str]) -> btypes.Material:
    """get a material from Python code and apply it to an object"""

    for path in paths:
        _update_sys_path(path)

    func_mod, func = func_desc.split('::')

    func_mod = importlib.import_module(func_mod)
    func = getattr(func_mod, func)

    return func(name, obj)


def build_add_node(mat: btypes.Material):
    """build a function to add a node"""
    def add_node(cls: type):
        """add a new node"""
        res = mat.node_tree.nodes.new(cls.__name__)
        return res
    return add_node


def add_links(mat: btypes.Material, links: List[Tuple]):
    """helper"""
    for link in links:
        blender.add_link(mat, *link)


def set_input(node: btypes.Node, name: Union[str, int], value: Any):
    """set default value of input for a node"""
    node.inputs[name].default_value = value


def set_output(node: btypes.Node, name: str, value: Any):
    """set default value of input for a node"""
    node.outputs[name].default_value = value
