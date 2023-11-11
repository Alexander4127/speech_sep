from collections import defaultdict
import glob
import logging
from pathlib import Path

from ss.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, *args, **kwargs):
        entries = defaultdict(dict)
        for path in glob.glob(f'{str(audio_dir)}/**', recursive=True):
            path = Path(path)
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                tp = path.parent.name.strip("s")
                assert tp in ["mix", "ref", "target"]

                file_name = path.name[:-len(path.suffix)]
                suffix = tp if tp != "mix" else "mixed"
                assert file_name.endswith(f"-{suffix}")
                idx = file_name[:-len(f"-{suffix}")]

                entries[idx][tp + "_path"] = path

        for entry in entries.values():
            entry["path"] = entry["text"] = ""
            entry["audio_len"] = entry["speaker_id"] = 0

        super().__init__(list(entries.values()), *args, **kwargs)
