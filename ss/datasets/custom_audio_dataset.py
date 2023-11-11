import logging
from pathlib import Path

import torchaudio

from ss.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            assert "mix" in entry and "target" in entry and "ref" in entry
            assert Path(entry["mix"]).exists(), f"Path {entry['mix']} doesn't exist"
            for key in ["mix", "target", "ref"]:
                entry[key] = str(Path(entry[key]).absolute().resolve())
            entry["text"] = entry.get("text", "")
            t_info = torchaudio.info(entry["mix"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(index, *args, **kwargs)
