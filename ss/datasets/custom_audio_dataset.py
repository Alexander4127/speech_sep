import logging
from pathlib import Path

import torchaudio

from ss.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            assert "mix_path" in entry and "target_path" in entry and "ref_path" in entry
            assert Path(entry["mix_path"]).exists(), f"Path {entry['mix_path']} doesn't exist"
            for key in ["mix_path", "target_path", "ref_path"]:
                entry[key] = str(Path(entry[key]).absolute().resolve())
            entry["text"] = entry.get("text", "")
            t_info = torchaudio.info(entry["mix_path"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, item):
        out = {}
        data_dict = self._index[item]
        out.update(data_dict)
        for t_name in ["mix", "target", "ref"]:
            cur_path = data_dict[t_name + "_path"]
            if t_name == "mix":
                mix_audio = self.load_audio(cur_path)
                out["aug_names"], out["mix"] = self.process_wave(mix_audio)
                continue
            out[t_name] = self.load_audio(cur_path)

        return out
