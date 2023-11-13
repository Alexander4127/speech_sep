from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from glob import glob
import json
import multiprocessing
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm
import typing as tp

from ss.base.base_dataset import BaseDataset
from ss.logger import logger
from ss.utils import ROOT_PATH
from .utils import create_mix


class NoisedDataset(BaseDataset):
    def __init__(self,
                 underlying: BaseDataset,
                 noise_dir: str,
                 name: str,
                 max_length: int = 1000,
                 reuse: bool = True,
                 snr_levels: tp.Tuple = (-5, -2, 0, 2, 5),
                 *args, **kwargs):
        data_dir = (ROOT_PATH / "data" / "datasets" / "noised").absolute().resolve()
        data_dir.mkdir(exist_ok=True, parents=True)
        assert isinstance(underlying, BaseDataset)

        noise_dir_path = Path(f'{ROOT_PATH}/{noise_dir}').absolute().resolve()
        self._noise_paths = [path for path in glob(f"{str(noise_dir_path)}/*.wav")]

        self._name = name
        self._index_dir = ROOT_PATH / "ss" / "datasets"
        self._data_dir = Path(data_dir)
        self._max_length = max_length
        self._snr_levels = [snr_levels] if not isinstance(snr_levels, tp.Iterable) else list(snr_levels)

        index = self._get_or_load_index(name, underlying, reuse)

        super().__init__(index, *args, **kwargs)
        self._assert_index_is_valid(self._index)

    def _get_or_load_index(self, name: str, underlying: BaseDataset, reuse: bool):
        index_path = self._index_dir / f"noised-{name}-index.json"
        if index_path.exists() and reuse:
            logger.warning('Reuse parameter set to True, reusing existing index.')
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(underlying)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _generate_triplets(self, underlying: BaseDataset):
        random.seed(0)
        logger.info(f'Len of under index {len(underlying._index)}')
        under_index = deepcopy(underlying._index)
        for idx, data in enumerate(under_index):
            data["index"] = idx

        all_triplets = {"ref": [], "target": [], "noise": [], "text": [], "path": [],
                        "speaker_id": [], "target_id": [], "noise_id": []}
        for _ in tqdm(range(self._max_length), desc="Preparing triplets..."):
            target = random.choice(under_index)
            noise_id = random.choice(range(len(self._noise_paths)))
            noise_path = self._noise_paths[noise_id]

            all_triplets["ref"].append(target["ref_path"])
            all_triplets["target"].append(target["mix_path"])
            all_triplets["text"].append(target["text"])
            all_triplets["path"].append(target["path"])
            all_triplets["noise"].append(noise_path)
            all_triplets["target_id"].append(target["index"])
            all_triplets["noise_id"].append(noise_id)

        return all_triplets

    def generate_mixes(self, underlying):
        triplets: tp.Dict[str, list] = self._generate_triplets(underlying)
        assert len(triplets["target"]) == self._max_length

        index = []
        out_dir = self._data_dir / "data" / self._name
        out_dir.mkdir(exist_ok=True, parents=True)
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
            futures = []
            for i in range(self._max_length):
                triplet = {"ref": triplets["ref"][i],
                           "target": triplets["target"][i],
                           "noise": triplets["noise"][i],
                           "target_id": triplets["target_id"][i],
                           "noise_id": triplets["noise_id"][i]}

                futures.append(pool.submit(create_mix, i, triplet, self._snr_levels, out_dir, test=True))

            for i, future in tqdm(enumerate(futures), desc="Creating mixes...", total=len(futures)):
                d_paths = future.result()
                if d_paths is None:
                    continue
                for d in d_paths:
                    d.update({"path": triplets["path"][i], "text": triplets["text"][i], "speaker_id": 0})
                    index.append(d)

        logger.info(f'Extracted {len(index)} audio')

        return index

    def _create_index(self, underlying: BaseDataset):
        return self.generate_mixes(underlying)

    def _assert_index_is_valid(self, index):
        for entry in index:
            assert "ref_path" in entry and "target_path" in entry and "mix_path" in entry, (
                "Each dataset item should include fields `ref_path`, `target_path` and `mix_path` - "
                "paths to reference, target and mixed audio respectively."
            )
            assert "speaker_id" in entry, "Each dataset item should include id of main speaker"

    def __getitem__(self, ind):
        out = {}
        data_dict = self._index[ind]
        out.update(data_dict)
        for t_name in ["mix", "target", "ref"]:
            cur_path = data_dict[t_name + "_path"]
            if t_name == "mix":
                mix_audio = self.load_audio(cur_path)
                out["aug_names"], out["mix"] = self.process_wave(mix_audio)
                continue
            out[t_name] = self.load_audio(cur_path)

        return out
