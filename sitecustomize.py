"""Runtime compatibility shims for mmcv2 + mmengine environments.

This repository is written against mmcv1-style APIs. When only mmcv2 is
available, imports such as `mmcv.runner`/`mmcv.parallel` fail very early.
The shim below provides the minimal mmcv1-compatible surface used by this
codebase, without changing installed torch/mmcv packages.
"""

from __future__ import annotations

import importlib
import sys
import types


def _identity_decorator(*_args, **_kwargs):
    def _wrap(func):
        return func

    return _wrap


def _cast_tensor_type(inputs, src_type=None, dst_type=None):
    import torch

    if isinstance(inputs, torch.Tensor):
        if src_type is None or inputs.dtype == src_type:
            return inputs.to(dst_type) if dst_type is not None else inputs
        return inputs
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(_cast_tensor_type(x, src_type, dst_type) for x in inputs)
    if isinstance(inputs, dict):
        return {k: _cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()}
    return inputs


def _obj_from_dict(info, parent=None, default_args=None):
    if not isinstance(info, dict):
        raise TypeError("info must be a dict")
    args = info.copy()
    obj_type = args.pop("type")
    if default_args is not None:
        for k, v in default_args.items():
            args.setdefault(k, v)
    if isinstance(obj_type, str):
        cls = getattr(parent, obj_type) if parent is not None else None
        if cls is None:
            raise KeyError(f"Cannot find type: {obj_type}")
    else:
        cls = obj_type
    return cls(**args)


def _build_optimizer(model, cfg):
    import torch

    cfg = cfg.copy()
    optim_type = cfg.pop("type")
    optim_cls = getattr(torch.optim, optim_type) if isinstance(optim_type, str) else optim_type
    return optim_cls(model.parameters(), **cfg)


class _OptimizerHook:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Fp16OptimizerHook(_OptimizerHook):
    pass


class _SimpleRunner:
    def __init__(self, model, optimizer, work_dir=None, logger=None, meta=None, max_epochs=1, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.logger = logger
        self.meta = meta
        self.max_epochs = max_epochs
        self.epoch = 0
        self.iter = 0
        self.timestamp = None
        self._hooks = []
        self.log_interval = 50

    def register_hook(self, hook, priority="NORMAL"):
        self._hooks.append(hook)

    def _call_hook(self, fn_name, *args, **kwargs):
        for hook in self._hooks:
            fn = getattr(hook, fn_name, None)
            if fn is not None:
                try:
                    fn(self, *args, **kwargs)
                except TypeError:
                    # Backward/forward-compatible fallback for legacy hook
                    # signatures that only accept runner.
                    fn(self)

    def register_training_hooks(
        self,
        lr_config,
        optimizer_config,
        checkpoint_config,
        log_config,
        momentum_config=None,
        custom_hooks_config=None,
    ):
        if isinstance(log_config, dict):
            self.log_interval = int(log_config.get("interval", 50))
        if custom_hooks_config:
            from mmengine.registry import HOOKS

            for cfg in custom_hooks_config:
                cfg = cfg.copy()
                priority = cfg.pop("priority", "NORMAL")
                hook = HOOKS.build(cfg)
                self.register_hook(hook, priority=priority)

    def resume(self, checkpoint):
        from mmengine.runner import load_checkpoint

        load_checkpoint(self.model, checkpoint, map_location="cpu")

    def load_checkpoint(self, checkpoint, revise_keys=None):
        from mmengine.runner import load_checkpoint

        load_checkpoint(self.model, checkpoint, map_location="cpu", revise_keys=revise_keys)

    def run(self, data_loaders, workflow):
        import torch
        from contextlib import nullcontext

        assert data_loaders, "data_loaders must not be empty"
        train_loader = data_loaders[0]
        total_iters = len(train_loader)
        self._call_hook("before_run")

        class _LegacyOptimWrapper:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def optim_context(self, model):
                return nullcontext()

            def update_params(self, loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        optim_wrapper = _LegacyOptimWrapper(self.optimizer)
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self._call_hook("before_train_epoch")
            self.model.train()
            for i, data_batch in enumerate(train_loader):
                self.iter = epoch * max(total_iters, 1) + i
                self._call_hook("before_train_iter", i, data_batch=data_batch)
                outputs = self.model.train_step(data_batch, optim_wrapper)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs

                if i % self.log_interval == 0:
                    if isinstance(outputs, dict) and "log_vars" in outputs:
                        log_vars = outputs["log_vars"]
                        if "loss" in log_vars:
                            print(f"Epoch [{epoch + 1}][{i + 1}/{total_iters}] loss: {float(log_vars['loss']):.6f}")
                        else:
                            show_keys = [k for k in log_vars.keys() if "loss" in k][:3]
                            if show_keys:
                                msg = ", ".join(f"{k}: {float(log_vars[k]):.6f}" for k in show_keys)
                                print(f"Epoch [{epoch + 1}][{i + 1}/{total_iters}] {msg}")
                            else:
                                print(f"Epoch [{epoch + 1}][{i + 1}/{total_iters}] iter done")
                    elif isinstance(loss, torch.Tensor):
                        print(f"Epoch [{epoch + 1}][{i + 1}/{total_iters}] loss: {float(loss.detach().cpu()):.6f}")
                self._call_hook("after_train_iter", i, data_batch=data_batch, outputs=outputs)
            self._call_hook("after_train_epoch")
        self._call_hook("after_run")


def _build_runner(cfg, default_args=None):
    cfg = cfg.copy()
    runner_type = cfg.pop("type", "EpochBasedRunner")
    if runner_type not in ("EpochBasedRunner", "IterBasedRunner"):
        raise NotImplementedError(f"Unsupported runner type in compatibility shim: {runner_type}")
    default_args = default_args or {}
    max_epochs = cfg.pop("max_epochs", 1)
    return _SimpleRunner(max_epochs=max_epochs, **default_args, **cfg)


def _install_mmcv_runner_shim():
    import torch
    from mmengine.dist import get_dist_info, init_dist, master_only
    from mmengine.hooks import DistSamplerSeedHook, Hook
    from mmengine.model import BaseModule, ModuleList, Sequential
    from mmengine.registry import HOOKS
    from mmengine.runner import load_checkpoint, load_state_dict

    runner_mod = types.ModuleType("mmcv.runner")
    runner_mod.get_dist_info = get_dist_info
    runner_mod.init_dist = init_dist
    runner_mod.load_checkpoint = load_checkpoint
    runner_mod.load_state_dict = load_state_dict
    runner_mod._load_checkpoint = lambda filename, map_location="cpu", logger=None: torch.load(
        filename, map_location=map_location
    )
    runner_mod.auto_fp16 = _identity_decorator
    runner_mod.force_fp32 = _identity_decorator
    runner_mod.BaseModule = BaseModule
    runner_mod.ModuleList = ModuleList
    runner_mod.Sequential = Sequential
    runner_mod.HOOKS = HOOKS
    runner_mod.Hook = Hook
    runner_mod.DistSamplerSeedHook = DistSamplerSeedHook
    runner_mod.OptimizerHook = _OptimizerHook
    runner_mod.Fp16OptimizerHook = _Fp16OptimizerHook
    runner_mod.BaseRunner = _SimpleRunner
    runner_mod.EpochBasedRunner = _SimpleRunner
    runner_mod.IterBasedRunner = _SimpleRunner
    runner_mod.build_optimizer = _build_optimizer
    runner_mod.build_runner = _build_runner
    runner_mod.obj_from_dict = _obj_from_dict

    base_module_mod = types.ModuleType("mmcv.runner.base_module")
    base_module_mod.BaseModule = BaseModule
    base_module_mod.ModuleList = ModuleList
    base_module_mod.Sequential = Sequential

    fp16_mod = types.ModuleType("mmcv.runner.fp16_utils")
    fp16_mod.cast_tensor_type = _cast_tensor_type

    hooks_mod = types.ModuleType("mmcv.runner.hooks")
    hooks_mod.HOOKS = HOOKS
    hooks_mod.Hook = Hook

    dist_mod = types.ModuleType("mmcv.runner.dist_utils")
    dist_mod.master_only = master_only

    sys.modules["mmcv.runner"] = runner_mod
    sys.modules["mmcv.runner.base_module"] = base_module_mod
    sys.modules["mmcv.runner.fp16_utils"] = fp16_mod
    sys.modules["mmcv.runner.hooks"] = hooks_mod
    sys.modules["mmcv.runner.dist_utils"] = dist_mod


def _install_mmcv_parallel_shim():
    import torch
    from mmengine.model import MMDistributedDataParallel, is_model_wrapper
    from torch.utils.data._utils.collate import default_collate

    class _MMDataParallel(torch.nn.Module):
        def __init__(self, module, device_ids=None):
            super().__init__()
            self.module = module
            if torch.cuda.is_available():
                dev = 0 if not device_ids else int(device_ids[0])
                self.module = module.cuda(dev)

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def train_step(self, *args, **kwargs):
            return self.module.train_step(*args, **kwargs)

        def val_step(self, *args, **kwargs):
            return self.module.val_step(*args, **kwargs)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    def _scatter(data, target_gpus, dim=0):
        # Minimal single-device scatter used by old mmcv inference utilities.
        return [data]

    class _DataContainer:
        def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
            self.data = data
            self._data = data
            self.stack = stack
            self.padding_value = padding_value
            self.cpu_only = cpu_only

    def _collate(batch, samples_per_gpu=1):
        if not isinstance(batch, list):
            return batch
        elem = batch[0]
        if isinstance(elem, _DataContainer):
            data_list = [sample.data for sample in batch]
            if elem.cpu_only:
                return data_list
            if elem.stack:
                return default_collate(data_list)
            return data_list
        if isinstance(elem, dict):
            return {k: _collate([d[k] for d in batch], samples_per_gpu) for k in elem}
        if isinstance(elem, tuple):
            transposed = list(zip(*batch))
            return tuple(_collate(list(items), samples_per_gpu) for items in transposed)
        if isinstance(elem, list):
            transposed = list(zip(*batch))
            return [_collate(list(items), samples_per_gpu) for items in transposed]
        return default_collate(batch)

    parallel_mod = types.ModuleType("mmcv.parallel")
    parallel_mod.MMDataParallel = _MMDataParallel
    parallel_mod.MMDistributedDataParallel = MMDistributedDataParallel
    parallel_mod.is_module_wrapper = is_model_wrapper
    parallel_mod.DataContainer = _DataContainer
    parallel_mod.collate = _collate
    parallel_mod.scatter = _scatter
    sys.modules["mmcv.parallel"] = parallel_mod


def _install_mmcv_configdict_shim():
    try:
        import mmcv
        import mmcv.utils as mmcv_utils
        import mmcv.cnn as mmcv_cnn
        from mmengine.config import Config, ConfigDict
        from mmengine.logging import print_log
        from mmengine.registry import Registry, build_from_cfg

        if not hasattr(mmcv, "ConfigDict"):
            mmcv.ConfigDict = ConfigDict
        if not hasattr(mmcv, "is_tuple_of"):
            def _is_tuple_of(obj, expected_type, length=None):
                if not isinstance(obj, tuple):
                    return False
                if length is not None and len(obj) != length:
                    return False
                return all(isinstance(item, expected_type) for item in obj)
            mmcv.is_tuple_of = _is_tuple_of
        if not hasattr(mmcv, "mkdir_or_exist"):
            import os
            mmcv.mkdir_or_exist = lambda path: os.makedirs(path, exist_ok=True)
        if not hasattr(mmcv, "load"):
            import pickle
            import json

            def _mmcv_load(file, file_format=None):
                if hasattr(file, "read"):
                    if file_format in (None, "pkl", "pickle"):
                        return pickle.load(file)
                    if file_format == "json":
                        return json.load(file)
                    raise ValueError(f"Unsupported file format: {file_format}")
                path = str(file)
                fmt = file_format or path.split(".")[-1]
                if fmt in ("pkl", "pickle"):
                    with open(path, "rb") as f:
                        return pickle.load(f)
                if fmt == "json":
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                raise ValueError(f"Unsupported file format: {fmt}")

            mmcv.load = _mmcv_load
        if not hasattr(mmcv, "dump"):
            import pickle
            import json

            def _mmcv_dump(obj, file, file_format=None):
                if hasattr(file, "write"):
                    if file_format in (None, "pkl", "pickle"):
                        return pickle.dump(obj, file)
                    if file_format == "json":
                        return json.dump(obj, file)
                    raise ValueError(f"Unsupported file format: {file_format}")
                path = str(file)
                fmt = file_format or path.split(".")[-1]
                if fmt in ("pkl", "pickle"):
                    with open(path, "wb") as f:
                        return pickle.dump(obj, f)
                if fmt == "json":
                    with open(path, "w", encoding="utf-8") as f:
                        return json.dump(obj, f)
                raise ValueError(f"Unsupported file format: {fmt}")

            mmcv.dump = _mmcv_dump
        if not hasattr(mmcv, "is_list_of"):
            def _is_list_of(obj, expected_type):
                return isinstance(obj, list) and all(isinstance(item, expected_type) for item in obj)
            mmcv.is_list_of = _is_list_of
        if not hasattr(mmcv, "FileClient"):
            from contextlib import contextmanager

            class _FileClient:
                def __init__(self, backend="disk", **kwargs):
                    self.backend = backend

                def get(self, filepath):
                    with open(filepath, "rb") as f:
                        return f.read()

                def get_text(self, filepath, encoding="utf-8"):
                    with open(filepath, "r", encoding=encoding) as f:
                        return f.read()

                @contextmanager
                def get_local_path(self, filepath):
                    yield filepath

            mmcv.FileClient = _FileClient
        if not hasattr(mmcv_utils, "ConfigDict"):
            mmcv_utils.ConfigDict = ConfigDict
        if not hasattr(mmcv_utils, "Config"):
            mmcv_utils.Config = Config
        if not hasattr(mmcv_utils, "Registry"):
            mmcv_utils.Registry = Registry
        if not hasattr(mmcv_utils, "build_from_cfg"):
            mmcv_utils.build_from_cfg = build_from_cfg
        if not hasattr(mmcv_utils, "print_log"):
            mmcv_utils.print_log = print_log
        if not hasattr(mmcv_utils, "get_git_hash"):
            mmcv_utils.get_git_hash = lambda: 'unknown'
        if not hasattr(mmcv_utils, "get_logger"):
            def _get_logger(name='mmcv', log_file=None, log_level='INFO'):
                import logging
                logger = logging.getLogger(name)
                if not logger.handlers:
                    logger.addHandler(logging.StreamHandler())
                logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
                return logger
            mmcv_utils.get_logger = _get_logger
        if not hasattr(mmcv_cnn, "NORM_LAYERS"):
            mmcv_cnn.NORM_LAYERS = Registry("norm layer")
        if not hasattr(mmcv_cnn, "MODELS"):
            mmcv_cnn.MODELS = Registry("models")
        if not hasattr(mmcv_cnn, "CONV_LAYERS"):
            mmcv_cnn.CONV_LAYERS = Registry("conv layer")
        if not hasattr(mmcv_cnn, "PLUGIN_LAYERS"):
            mmcv_cnn.PLUGIN_LAYERS = Registry("plugin layer")
        if not hasattr(mmcv_cnn, "UPSAMPLE_LAYERS"):
            mmcv_cnn.UPSAMPLE_LAYERS = Registry("upsample layer")
        if not hasattr(mmcv_cnn, "ACTIVATION_LAYERS"):
            mmcv_cnn.ACTIVATION_LAYERS = Registry("activation layer")
        if not hasattr(mmcv_cnn, "constant_init"):
            def _constant_init(module, val, bias=0):
                import torch.nn as nn
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, val)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, bias)
            mmcv_cnn.constant_init = _constant_init
        if not hasattr(mmcv_cnn, "trunc_normal_init"):
            def _trunc_normal_init(module, mean=0, std=1, a=-2, b=2, bias=0):
                import torch.nn as nn
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, bias)
            mmcv_cnn.trunc_normal_init = _trunc_normal_init
        if not hasattr(mmcv_cnn, "xavier_init"):
            def _xavier_init(module, gain=1, bias=0, distribution='normal'):
                import torch.nn as nn
                if hasattr(module, 'weight') and module.weight is not None:
                    if distribution == 'uniform':
                        nn.init.xavier_uniform_(module.weight, gain=gain)
                    else:
                        nn.init.xavier_normal_(module.weight, gain=gain)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, bias)
            mmcv_cnn.xavier_init = _xavier_init
        if not hasattr(mmcv_cnn, "bias_init_with_prob"):
            import math
            mmcv_cnn.bias_init_with_prob = lambda p: float(-math.log((1 - p) / p))
        if "mmcv.cnn.utils.weight_init" not in sys.modules:
            weight_init_mod = types.ModuleType("mmcv.cnn.utils.weight_init")
            weight_init_mod.constant_init = mmcv_cnn.constant_init
            weight_init_mod.trunc_normal_init = mmcv_cnn.trunc_normal_init
            weight_init_mod.xavier_init = mmcv_cnn.xavier_init
            sys.modules["mmcv.cnn.utils.weight_init"] = weight_init_mod
        if "mmcv.cnn.bricks.registry" not in sys.modules:
            bricks_reg_mod = types.ModuleType("mmcv.cnn.bricks.registry")
            bricks_reg_mod.CONV_LAYERS = mmcv_cnn.CONV_LAYERS
            bricks_reg_mod.NORM_LAYERS = mmcv_cnn.NORM_LAYERS
            bricks_reg_mod.PLUGIN_LAYERS = mmcv_cnn.PLUGIN_LAYERS
            bricks_reg_mod.UPSAMPLE_LAYERS = mmcv_cnn.UPSAMPLE_LAYERS
            bricks_reg_mod.ACTIVATION_LAYERS = mmcv_cnn.ACTIVATION_LAYERS
            bricks_reg_mod.ATTENTION = Registry("attention")
            bricks_reg_mod.TRANSFORMER_LAYER = Registry("transformer layer")
            bricks_reg_mod.TRANSFORMER_LAYER_SEQUENCE = Registry("transformer layer sequence")
            sys.modules["mmcv.cnn.bricks.registry"] = bricks_reg_mod
    except Exception:
        pass


def _patch():
    try:
        importlib.import_module("mmcv")
    except Exception:
        return

    try:
        importlib.import_module("mmcv.runner")
    except Exception:
        _install_mmcv_runner_shim()

    try:
        importlib.import_module("mmcv.parallel")
    except Exception:
        _install_mmcv_parallel_shim()

    _install_mmcv_configdict_shim()
    _install_mmdet_legacy_shims()


def _install_mmdet_legacy_shims():
    try:
        import mmdet
        from mmdet import registry as mmdet_registry
    except Exception:
        return

    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    def _legacy_build_dataloader(
        dataset,
        samples_per_gpu,
        workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=True,
        seed=None,
        runner_type="EpochBasedRunner",
        persistent_workers=False,
        **kwargs,
    ):
        if dist:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            batch_size = samples_per_gpu
            shuffle = False
        else:
            sampler = None
            batch_size = samples_per_gpu * max(1, num_gpus)
        from mmcv.parallel import collate as mmcv_collate
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            collate_fn=lambda x: mmcv_collate(x, samples_per_gpu=samples_per_gpu),
            num_workers=workers_per_gpu,
            pin_memory=False,
            drop_last=False,
            persistent_workers=persistent_workers,
        )

    def _multi_apply(func, *args, **kwargs):
        pfunc = (lambda *a: func(*a, **kwargs)) if kwargs else func
        map_results = map(pfunc, *args)
        return tuple(map(list, zip(*map_results)))

    def _reduce_mean(tensor):
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt

    class _EvalHook:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _DistEvalHook(_EvalHook):
        pass

    # mmdet.models.builder
    if "mmdet.models.builder" not in sys.modules:
        builder_mod = types.ModuleType("mmdet.models.builder")
        builder_mod.BACKBONES = mmdet_registry.MODELS
        builder_mod.DETECTORS = mmdet_registry.MODELS
        builder_mod.HEADS = mmdet_registry.MODELS
        builder_mod.LOSSES = mmdet_registry.MODELS
        builder_mod.NECKS = mmdet_registry.MODELS
        builder_mod.ROI_EXTRACTORS = mmdet_registry.MODELS
        builder_mod.SHARED_HEADS = mmdet_registry.MODELS
        builder_mod.build_backbone = mmdet_registry.MODELS.build
        builder_mod.build_detector = mmdet_registry.MODELS.build
        builder_mod.build_head = mmdet_registry.MODELS.build
        builder_mod.build_loss = mmdet_registry.MODELS.build
        builder_mod.build_neck = mmdet_registry.MODELS.build
        sys.modules["mmdet.models.builder"] = builder_mod
    else:
        builder_mod = sys.modules["mmdet.models.builder"]

    # mmdet.models convenience attrs expected by mmdet2-style code
    try:
        import mmdet.models as mmdet_models
        for name in ["BACKBONES", "DETECTORS", "HEADS", "LOSSES", "NECKS",
                     "ROI_EXTRACTORS", "SHARED_HEADS"]:
            if not hasattr(mmdet_models, name):
                setattr(mmdet_models, name, getattr(builder_mod, name))
    except Exception:
        pass

    # mmdet.models.utils.builder
    if "mmdet.models.utils.builder" not in sys.modules:
        model_utils_builder = types.ModuleType("mmdet.models.utils.builder")
        model_utils_builder.TRANSFORMER = mmdet_registry.MODELS
        sys.modules["mmdet.models.utils.builder"] = model_utils_builder

    # mmdet.models.utils
    try:
        import mmdet.models.utils as model_utils
    except Exception:
        model_utils = types.ModuleType("mmdet.models.utils")
        sys.modules["mmdet.models.utils"] = model_utils
    if not hasattr(model_utils, "build_transformer"):
        model_utils.build_transformer = mmdet_registry.MODELS.build

    # mmdet.core
    if "mmdet.core" not in sys.modules:
        core_mod = types.ModuleType("mmdet.core")
        core_mod.multi_apply = _multi_apply
        core_mod.reduce_mean = _reduce_mean
        core_mod.EvalHook = _EvalHook
        core_mod.DistEvalHook = _DistEvalHook
        sys.modules["mmdet.core"] = core_mod

    # mmdet.datasets.builder
    if "mmdet.datasets.builder" not in sys.modules:
        ds_builder = types.ModuleType("mmdet.datasets.builder")
        ds_builder.DATASETS = mmdet_registry.DATASETS
        ds_builder.PIPELINES = mmdet_registry.TRANSFORMS
        ds_builder.build_dataloader = _legacy_build_dataloader
        ds_builder._concat_dataset = lambda cfg, default_args=None: None
        sys.modules["mmdet.datasets.builder"] = ds_builder

    # mmdet.datasets convenience attrs
    try:
        import mmdet.datasets as mmdet_datasets
        if not hasattr(mmdet_datasets, "DATASETS"):
            mmdet_datasets.DATASETS = mmdet_registry.DATASETS
        if not hasattr(mmdet_datasets, "PIPELINES"):
            mmdet_datasets.PIPELINES = mmdet_registry.TRANSFORMS
        if not hasattr(mmdet_datasets, "build_dataloader"):
            mmdet_datasets.build_dataloader = _legacy_build_dataloader
        if not hasattr(mmdet_datasets, "replace_ImageToTensor"):
            mmdet_datasets.replace_ImageToTensor = lambda pipeline: pipeline
    except Exception:
        pass

    # mmdet.datasets.pipelines
    if "mmdet.datasets.pipelines" not in sys.modules:
        pipes_mod = types.ModuleType("mmdet.datasets.pipelines")
        try:
            from mmdet.datasets.transforms import LoadAnnotations, RandomCrop, RandomFlip
        except Exception:
            LoadAnnotations = object
            RandomCrop = object
            RandomFlip = object

        class _LoadImageFromFile:
            def __call__(self, results):
                return results

        def _to_tensor(data):
            return torch.as_tensor(data)

        pipes_mod.LoadAnnotations = LoadAnnotations
        pipes_mod.LoadImageFromFile = _LoadImageFromFile
        pipes_mod.RandomCrop = RandomCrop
        pipes_mod.RandomFlip = RandomFlip
        pipes_mod.Rotate = object
        pipes_mod.MultiScaleFlipAug = object
        pipes_mod.to_tensor = _to_tensor
        sys.modules["mmdet.datasets.pipelines"] = pipes_mod

    # mmdet.apis.set_random_seed
    try:
        import mmdet.apis as mmdet_apis

        if not hasattr(mmdet_apis, "set_random_seed"):
            def _set_random_seed(seed, deterministic=False):
                import random
                import numpy as np

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if deterministic:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

            mmdet_apis.set_random_seed = _set_random_seed
    except Exception:
        pass


_patch()
