# Copyright (c) OpenMMLab. All rights reserved.
#
# NOTE:
# This package is used under a mmcv2/mmengine stack via compatibility shims.
# Keep top-level imports minimal to avoid importing unused legacy modules that
# depend on removed mmdet2 namespaces.

from .bbox.structures import (  # noqa: F401
    Box3DMode,
    CameraInstance3DBoxes,
    Coord3DMode,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    get_box_type,
)
from .bbox.transforms import bbox3d2result  # noqa: F401
from .hook import *  # noqa: F401, F403
from .points import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .voxel import *  # noqa: F401, F403


def show_result(*args, **kwargs):
    return None


def merge_aug_bboxes_3d(aug_results, *args, **kwargs):
    return aug_results[0] if isinstance(aug_results, (list, tuple)) and aug_results else aug_results
