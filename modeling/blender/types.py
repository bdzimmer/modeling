"""
Types for Blender interface.
"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from typing import Optional, Dict, List


class ConfigTypes:
    """constants"""
    MODEL = 'ConfigModel'
    INSTANCE = 'ConfigInstance'
    EMPTY = 'ConfigEmpty'
    LIGHT = 'ConfigLight'
    ARMATURE = 'ConfigArmature'
    BONE = 'ConfigBone'


class ConfigObject:
    """model configuration"""

    def __init__(
            self,
            name: str,
            transformation: Optional[Dict] = None,
            collection: Optional[str] = None,
            hide: bool = False,
            props: Optional[List[Dict]] = None,
            children: Optional[List] = None
            ):
        """constructor"""

        self.name = name
        self.transformation = transformation
        self.collection = collection
        self.hide = hide
        self.props = props
        self.children = children


class ConfigEmpty(ConfigObject):
    """model configuration"""

    def __init__(
            self,
            name: str,
            transformation: Optional[Dict] = None,
            collection: Optional[str] = None,
            hide: bool = False,
            props: Optional[List[Dict]] = None,
            children: Optional[List] = None
            ):
        """constructor"""

        super().__init__(
            name=name,
            transformation=transformation,
            collection=collection,
            hide=hide,
            props=props,
            children=children
        )


# TODO: implement ConfigLight properly

class ConfigLight(ConfigObject):
    """model configuration"""

    def __init__(
            self,
            name: str,
            transformation: Optional[Dict] = None,
            collection: Optional[str] = None,
            hide: bool = False,
            props: Optional[List[Dict]] = None,
            children: Optional[List] = None
            ):
        """constructor"""

        super().__init__(
            name=name,
            transformation=transformation,
            collection=collection,
            hide=hide,
            props=props,
            children=children
        )


class ConfigModel(ConfigObject):
    """model configuration"""

    def __init__(
            self,
            name: str,
            transformation: Optional[Dict] = None,
            collection: Optional[str] = None,
            hide: bool = False,
            props: Optional[List[Dict]] = None,
            children: Optional[List] = None,

            filename: Optional[str] = None,
            color: Optional[List] = None,
            material: Optional[Dict] = None,
            auto_smooth_angle: Optional[float] = None
            ):
        """constructor"""

        super().__init__(
            name=name,
            transformation=transformation,
            collection=collection,
            hide=hide,
            props=props,
            children=children
        )

        self.filename = filename
        self.color = color
        self.material = material

        self.auto_smooth_angle = auto_smooth_angle
        self.hide = hide


class ConfigInstance(ConfigObject):
    """model configuration"""

    def __init__(
            self,
            name: str,
            transformation: Optional[Dict] = None,
            collection: Optional[str] = None,
            hide: bool = False,
            props: Optional[List[Dict]] = None,
            children: Optional[List] = None,

            instance: Optional[str] = None,
            color: Optional[List] = None,
            material: Optional[Dict] = None,
            auto_smooth_angle: Optional[float] = None
            ):
        """constructor"""

        super().__init__(
            name=name,
            transformation=transformation,
            collection=collection,
            hide=hide,
            props=props,
            children=children
        )

        self.instance = instance
        self.color = color
        self.material = material

        # TODO: this probably isn't necessary
        self.auto_smooth_angle = auto_smooth_angle


class ConfigArmature(ConfigObject):
    """model configuration"""

    def __init__(
            self,
            name: str,
            transformation: Optional[Dict] = None,
            collection: Optional[str] = None,
            hide: bool = False,
            props: Optional[List[Dict]] = None,
            children: Optional[List] = None
            ):
        """constructor"""

        super().__init__(
            name=name,
            transformation=transformation,
            collection=collection,
            hide=hide,
            props=props,
            children=children
        )


class ConfigBone(ConfigObject):
    """model configuration"""

    def __init__(
            self,
            name: str,

            head: List[float],
            tail: List[float],
            roll: float,
            parent_bone_name: Optional[str],  # parent bone
            connect: bool,                    # whether or not to directly connect
            child_mesh_name: Optional[str],   # child mesh

            transformation: Optional[Dict] = None,
            collection: Optional[str] = None,
            hide: bool = False,
            props: Optional[List[Dict]] = None,
            children: Optional[List] = None
            ):
        """constructor"""

        super().__init__(
            name=name,
            transformation=transformation,
            collection=collection,
            hide=hide,
            props=props,
            children=children
        )

        self.head = head
        self.tail = tail
        self.roll = roll
        self.parent_bone_name = parent_bone_name
        self.connect = connect
        self.child_mesh_name = child_mesh_name


class ModelKeys:
    """keys for model dict"""

    NAME = 'name'              # name of model
    CLASS = 'class'
    FILENAME = 'filename'      # filename to load, usually obj
    COPY = 'copy'              # name of other model to copy mesh data from
    COLOR = 'color'            # default / overall color for the model
    MATERIAL = 'material'      # material information
    COLLECTION = 'collection'  # collection name
    INFO = 'info'

    # if present, enable auto smoothing at the specified angle
    AUTO_SMOOTH_ANGLE = 'auto_smooth_angle'

    # dict with 'rotation', 'translation', and 'scale' fields.
    TRANSFORMATION = 'transformation'

    HIDE = 'hide'                                  # if true, hide the model
    PROPS = 'props'                                # optional properties, such as modifiers

    INSTANCE = 'instance'

    # a list of additional model objects that should be positioned
    # relative to this one
    CHILDREN = 'children'


class PropsKeys:
    """keys for properties"""

    VALUE = 'value'

    AUTO_SMOOTH_ANGLE = 'auto_smooth_angle'
    AUTO_SMOOTH_ANGLE_VALUE = 'value'

    BLENDER_TRIS_TO_QUADS = 'blender:tris_to_quads'

    BLENDER_WIREFRAME = 'blender:wireframe'
    BLENDER_WIREFRAME_THICKNESS = 'thickness'
    BLENDER_WIREFRAME_USE_EVEN_OFFSET = 'use_even_offset'

    BLENDER_SHADOW_DISABLE = 'blender:shadow_disable'

    BLENDER_EMISSION_STRENGTH = 'blender:emission_strength'

    BLENDER_ALPHA = 'blender:alpha'

    BLENDER_SUBSURFACE = 'blender:subsurface'
    BLENDER_SUBSURFACE_LEVELS = 'levels'
    BLENDER_SUBSURFACE_RENDER_LEVELS = 'render_levels'
    BLENDER_SUBSURFACE_USE_ADAPTIVE_SUBDIVISION = 'use_adaptive_subdivision'
    BLENDER_BEVEL = 'blender:bevel'
    BLENDER_BEVEL_WIDTH = 'width'
    BLENDER_CHILD_OF = "blender:child_of"


class TransformationKeys:
    """keys for transformations"""
    ROTATION = 'rotation'
    TRANSLATION = 'translation'
    SCALE = 'scale'


class KeyframeKeys:
    """keys for animation keyframes"""
    TRANSFORMATION = 'transformation'
    HIDE = 'hide'


class MaterialKeys:
    """keys for materials"""
    INSTANCE = 'instance'
    PYTHON = 'python'
    MATLIB = 'matlib'
    ASSETLIB = 'assetlib'

    PYTHON_FUNC = 'func'
    PYTHON_PATHS = 'paths'

    MATLIB_LIB_INDEX = 'lib_index'
    MATLIB_MAT_INDEX = 'mat_index'

    MATLIB_LIB_NAME = 'lib_name'
    MATLIB_MAT_NAME = 'mat_name'

    ASSETLIB_LIB_NAME = 'lib_name'
    ASSETLIB_MAT_NAME = 'mat_name'

    NAME = 'name'
    COPY = 'copy'
    COLOR = 'color'
    DIFFUSE = 'diffuse'
    EMISSION = 'emission'
    UPDATES = 'updates'
