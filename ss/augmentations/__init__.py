from collections.abc import Callable
from typing import List

import ss.augmentations.wave_augmentations
from ss.augmentations.random_apply import RandomApply
from ss.augmentations.sequential import SequentialAugmentation
from ss.utils.parse_config import ConfigParser


def from_configs(configs: ConfigParser):
    wave_augs = []
    if "augmentations" in configs.config and "wave" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["wave"]:
            wave_augs.append(
                configs.init_obj(aug_dict, ss.augmentations.wave_augmentations)
            )

    aug_prob = 0
    if "augmentations" in configs.config and "p" in configs.config["augmentations"]:
        aug_prob = configs.config["augmentations"]["p"]

    return _to_function(wave_augs, aug_prob)


def _to_function(augs_list: List[Callable], aug_prob: float):
    if len(augs_list) == 0:
        return None
    else:
        return SequentialAugmentation([RandomApply(aug, aug_prob) for aug in augs_list])
