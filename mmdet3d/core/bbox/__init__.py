# Copyright (c) OpenMMLab. All rights reserved.
#
# Keep bbox package imports lightweight for mmcv2/mmengine compatibility.

from .structures import (  # noqa: F401
    BaseInstance3DBoxes,
    Box3DMode,
    CameraInstance3DBoxes,
    Coord3DMode,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    get_box_type,
    limit_period,
    mono_cam_box2vis,
    points_cam2img,
    points_img2cam,
    xywhr2xyxyr,
)
from .transforms import bbox3d2result, bbox3d2roi, bbox3d_mapping_back  # noqa: F401

__all__ = [
    'Box3DMode', 'LiDARInstance3DBoxes', 'CameraInstance3DBoxes',
    'bbox3d2roi', 'bbox3d2result', 'DepthInstance3DBoxes',
    'BaseInstance3DBoxes', 'bbox3d_mapping_back', 'xywhr2xyxyr',
    'limit_period', 'points_cam2img', 'points_img2cam', 'get_box_type',
    'Coord3DMode', 'mono_cam_box2vis'
]
