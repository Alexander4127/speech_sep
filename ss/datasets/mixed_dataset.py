from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from glob import glob
from itertools import groupby
import json
import multiprocessing
import os
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm
import typing as tp

import ss
from ss.base.base_dataset import BaseDataset
from ss.logger import logger
from ss.utils import ROOT_PATH
from ss.utils.parse_config import ConfigParser
from .utils import create_mix


class PathHolder:
    path: str
    id: int
    text: str

    def __init__(self, path: str, index: int, text: str):
        self.path = path
        self.id = index
        self.text = text


class MixedDataset(BaseDataset):
    def __init__(self,
                 underlying: BaseDataset,
                 name: str,
                 max_speakers: int = 100,
                 max_length: int = 1_000_000,
                 reuse: bool = True,
                 test: bool = False,
                 snr_levels: tp.Tuple = (-5, -2, 0, 2, 5),
                 vad_merge: tp.Optional[int] = 20,
                 *args, **kwargs):
        data_dir = ROOT_PATH / "data" / "datasets" / "mixed"
        data_dir.mkdir(exist_ok=True, parents=True)
        assert isinstance(underlying, BaseDataset)

        self._name = name
        self._index_dir = ROOT_PATH / "ss" / "datasets"
        self._data_dir = Path(data_dir)
        self._max_speakers = max_speakers
        self._max_length = max_length
        self._test = test
        self._snr_levels = [snr_levels] if not isinstance(snr_levels, tp.Iterable) else list(snr_levels)
        self._vad_merge = vad_merge

        index = self._get_or_load_index(name, underlying, reuse)
        self._speakers = sorted({d["speaker_id"] for d in index})
        assert len(self._speakers) == self._max_speakers, f'Expected {len(self._speakers)} = {self._max_speakers}'
        self._speakers: tp.Dict[int, int] = {spk_id: idx for idx, spk_id in enumerate(self._speakers)}

        super().__init__(index, *args, **kwargs)
        self._assert_index_is_valid(self._index)

    def _get_or_load_index(self, name: str, underlying: BaseDataset, reuse: bool):
        index_path = self._index_dir / f"mixed-{name}-index.json"
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
        under_index = deepcopy(underlying._index)
        for idx, data in enumerate(under_index):
            data["index"] = idx

        def get_speaker_id(d):
            return int(Path(d["path"]).name.split('-')[0])

        speaker_idx = set(get_speaker_id(d) for d in under_index)
        sorted_index = sorted(under_index, key=get_speaker_id)
        speaker_audio_paths = {k: [PathHolder(el["path"], el["index"], el["text"]) for el in g]
                               for k, g in groupby(sorted_index, key=get_speaker_id)}

        main_speakers = sorted(random.sample(list(speaker_idx), k=min(self._max_speakers, len(speaker_idx))))
        logger.info(f'Filtered {len(main_speakers)} main speakers from {len(speaker_idx)} and'
                    f' {sum([len(speaker_audio_paths[idx]) for idx in main_speakers])} main audio')

        all_triplets = {"ref": [], "target": [], "noise": [], "text": [],
                        "speaker_id": [], "target_id": [], "noise_id": []}
        for _ in tqdm(range(self._max_length), desc="Preparing triplets..."):
            main_speaker_idx = random.choice(main_speakers)
            second_speaker_idx = random.choice(list(speaker_idx - {main_speaker_idx}))

            target, reference = random.sample(speaker_audio_paths[main_speaker_idx], k=2)
            noise = random.choice(speaker_audio_paths[second_speaker_idx])
            all_triplets["ref"].append(reference.path)
            all_triplets["target"].append(target.path)
            all_triplets["text"].append(target.text)
            all_triplets["noise"].append(noise.path)
            all_triplets["speaker_id"].append(main_speaker_idx)
            all_triplets["target_id"].append(target.id)
            all_triplets["noise_id"].append(noise.id)

        return all_triplets

    def generate_mixes(self, underlying):
        triplets: tp.Dict[str, list] = self._generate_triplets(underlying)
        assert len(triplets["target"]) == self._max_length

        speaker_index = [
            {
                "speaker_id": triplets["speaker_id"][i],
                "path": triplets["target"][i],
                "text": triplets["text"][i]
            } for i in range(self._max_length)]
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

                futures.append(pool.submit(create_mix, i, triplet, self._snr_levels, out_dir,
                                           test=self._test, vad_db=self._vad_merge))

            for i, future in tqdm(enumerate(futures), desc="Creating mixes...", total=len(futures)):
                d_paths = future.result()
                if d_paths is None:
                    continue
                for d in d_paths:
                    d.update(speaker_index[i])
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

        out["speaker_id"] = self._speakers[data_dict["speaker_id"]]

        return out
