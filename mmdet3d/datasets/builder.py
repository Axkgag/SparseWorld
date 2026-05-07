# Copyright (c) OpenMMLab. All rights reserved.
import platform

from mmcv.utils import Registry, build_from_cfg

try:
    from mmdet.datasets import DATASETS as MMDET_DATASETS
except Exception:
    from mmdet.registry import DATASETS as MMDET_DATASETS
try:
    from mmdet.datasets.builder import _concat_dataset
except Exception:
    def _concat_dataset(cfg, default_args=None):
        from mmdet.datasets.dataset_wrappers import ConcatDataset
        datasets = []
        for ann_file in cfg['ann_file']:
            data_cfg = cfg.copy()
            data_cfg['ann_file'] = ann_file
            datasets.append(build_dataset(data_cfg, default_args))
        return ConcatDataset(datasets)

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    try:
        from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                     ConcatDataset, RepeatDataset)
    except Exception:
        from torch.utils.data import ConcatDataset
        ClassBalancedDataset = None
        RepeatDataset = None
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        if RepeatDataset is None:
            raise RuntimeError('RepeatDataset is unavailable in current mmdet version.')
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        if ClassBalancedDataset is None:
            raise RuntimeError('ClassBalancedDataset is unavailable in current mmdet version.')
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    elif cfg['type'] in DATASETS._module_dict.keys():
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    else:
        dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=False,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     persistent_workers=False,
                     **kwargs):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from mmcv.parallel import collate

    if dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        batch_size = samples_per_gpu
        shuffle = False
    else:
        sampler = None
        batch_size = samples_per_gpu * max(1, num_gpus)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=lambda x: collate(x, samples_per_gpu=samples_per_gpu),
        num_workers=workers_per_gpu,
        pin_memory=False,
        drop_last=False,
        persistent_workers=persistent_workers)
