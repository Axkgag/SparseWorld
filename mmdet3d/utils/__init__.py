# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg, print_log

try:
    from .collect_env import collect_env
except Exception:
    collect_env = lambda: {}
from .compat_cfg import compat_cfg
from .logger import get_root_logger
from .misc import find_latest_checkpoint
from .setup_env import setup_multi_processes
from .patch import patch_config, patch_runner, find_latest_checkpoint

__all__ = [
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env',
    'print_log', 'setup_multi_processes', 'find_latest_checkpoint',
    'compat_cfg',
    "patch_config",
    "patch_runner",
    "find_latest_checkpoint",
]


def register_all_modules(init_default_scope=True):
    # Compatibility entrypoint expected by MMEngine/MMDet3 auto-import logic.
    # Importing these packages triggers @register_module side effects.
    import mmdet3d.datasets  # noqa: F401
    import mmdet3d.models  # noqa: F401
