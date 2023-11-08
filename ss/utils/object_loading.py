import logging
from operator import xor

from torch.utils.data import DataLoader

import ss.augmentations
import ss.datasets
from ss import batch_sampler as batch_sampler_module
from ss.collate_fn.collate import collate_fn
from ss.utils.parse_config import ConfigParser


logger = logging.getLogger()


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            augs = ss.augmentations.from_configs(configs)
            drop_last = True
        else:
            augs = None
            drop_last = False

        # create and join datasets
        ds = params["dataset"]
        if ds["type"] == "MixedDataset":
            ds["args"]["name"] = ds["args"]["underlying"]["args"]["part"]
            ds["args"]["underlying"] = configs.init_obj(
                ds["args"]["underlying"], ss.datasets, config_parser=configs)
        dataset = configs.init_obj(ds, ss.datasets, config_parser=configs, augs=augs)

        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
