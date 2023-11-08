import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}

    # calculating max lengths of audio
    mix_lengths, ref_lengths = [], []
    for item in dataset_items:
        mix_lengths.append(max([item["mix"].shape[1], item["target"].shape[1]]))
        ref_lengths.append(item["ref"].shape[1])
    result_batch.update({"mix_lengths": torch.tensor(mix_lengths), "ref_lengths": torch.tensor(ref_lengths)})

    # padding and stacking to one tensor
    for key, batch_len in zip(["mix", "target", "ref"], [max(mix_lengths), max(mix_lengths), max(ref_lengths)]):
        result_batch[key] = torch.zeros(len(dataset_items), batch_len)
        for idx, item in enumerate(dataset_items):
            result_batch[key][idx, :item[key].shape[-1]] = item[key].squeeze()

    # converting speaker ids to tensor
    result_batch["speaker_id"] = torch.tensor([item["speaker_id"] for item in dataset_items])

    for k in set(dataset_items[0].keys()) - set(result_batch.keys()):
        result_batch[k] = [item[k] for item in dataset_items]

    return result_batch
