"""

Functions for loading materials into Blender.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

import os
import importlib
import sys
from typing import List, Tuple, Any, Union, Dict

import bpy
import bpy.types as btypes

from modeling.blender import util


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


def build_add_node(mat: Union[btypes.Material, btypes.Scene]):
    """build a function to add a node"""
    def add_node(cls: type):
        """add a new node"""
        res = mat.node_tree.nodes.new(cls.__name__)
        return res
    return add_node


def add_links(mat: btypes.Material, links: List[Tuple]):
    """helper"""
    for link in links:
        util.add_link(mat, *link)


def set_input(node: btypes.Node, name: Union[str, int], value: Any):
    """set default value of input for a node"""
    node.inputs[name].default_value = value


def set_output(node: btypes.Node, name: str, value: Any):
    """set default value of input for a node"""
    node.outputs[name].default_value = value


def matlib_select(lib_name: str, mat_name: str) -> None:
    """select a materials library material by name"""

    matlib = bpy.context.scene.matlib

    for idx, library in enumerate(matlib.libraries):
        if library.name == lib_name + '.blend':
            matlib.lib_index = idx
            break
    else:
        print(f'library {lib_name} not found')

    for idx, material in enumerate(matlib.materials):
        if material.name == mat_name:
            matlib.mat_index = idx
            break
    else:
        print(f'material {mat_name} not found')


def find_material_assetlib_blends(lib_name: str) -> Dict[str, str]:
    """find which blend files contain which materials for an asset library"""

    # https://blender.stackexchange.com/questions/244971/how-do-i-get-all-assets-in-a-given-userassetlibrary-with-the-python-api

    libs = bpy.context.preferences.filepaths.asset_libraries
    lib = [x for x in libs if x.name == lib_name][0]
    blend_file_names = [x for x in os.listdir(lib.path) if x.endswith('.blend')]
    res = {}
    for blend_file_name in blend_file_names:
        blend_file_path = os.path.join(lib.path, blend_file_name)
        with bpy.data.libraries.load(blend_file_path, assets_only=True) as (data_from, _):
            for mat_name in data_from.materials:
                res[mat_name] = blend_file_path
    return res


def find_material_assetlib(lib_name: str, mat_name: str) -> btypes.Material:
    """find a material in asset libraries"""

    # TODO: this seems pretty fast for a few materials but may be inefficient for many
    mat_name_to_path = find_material_assetlib_blends(lib_name)
    blend_file_path = mat_name_to_path[mat_name]

    # load and append the material
    with bpy.data.libraries.load(blend_file_path, assets_only=True) as (data_from, data_to):
        data_to.materials = [mat_name]
    mat = bpy.data.materials.get(mat_name)
    return mat
