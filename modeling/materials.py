"""

Functions for loading materials into Blender.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.


import importlib
import sys
from typing import List

import bpy.types


def _update_sys_path(path: str) -> None:
    """add modules to sys.path"""
    if path not in sys.path:
        print(f'adding {path} to sys.path')
        sys.path.append(path)


def material_python(
        name: str,
        obj: bpy.types.Object,
        func_desc: str,
        paths: List[str]) -> bpy.types.Material:
    """get a material from Python code and apply it to an object"""

    for path in paths:
        _update_sys_path(path)

    func_mod, func = func_desc.split('::')

    func_mod = importlib.import_module(func_mod)
    func = getattr(func_mod, func)

    # TODO: something needs to change here to allow the layouts to work properly

    return func(name, obj)
