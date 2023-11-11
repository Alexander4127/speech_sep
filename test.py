import argparse
import multiprocessing
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import torch
from tqdm import tqdm

from ss.metric import PESQMetric, SISDRMetric
import ss.model as module_model
from ss.trainer import Trainer
from ss.utils import ROOT_PATH
from ss.utils.object_loading import get_dataloaders
from ss.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file, gen_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # make dirs for denoised dataset
    if gen_dir is not None:
        gen_dir = Path(gen_dir)
        gen_dir.mkdir(exist_ok=True, parents=True)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = defaultdict(lambda: defaultdict(float))

    sr = config["preprocessing"]["sr"]
    pesq = PESQMetric(sr=sr)
    si_sdr = SISDRMetric()
    meter = pyln.Meter(sr=sr)
    with torch.no_grad():
        for test_name in filter(lambda name: name.startswith("test"), dataloaders.keys()):
            if gen_dir is not None:
                audio_dir = (gen_dir / test_name / "audio").absolute().resolve()
                audio_dir.mkdir(exist_ok=True, parents=True)
                text_dir = (gen_dir / test_name / "audio").absolute().resolve()
                text_dir.mkdir(exist_ok=True, parents=True)

            n_samples = 0
            for batch_num, batch in enumerate(tqdm(dataloaders[test_name], desc=test_name)):
                batch = Trainer.move_batch_to_device(batch, device)
                output = model(**batch)
                batch.update(output)

                for i in range(len(batch["short"])):
                    loudness = meter.integrated_loudness(batch["short"])
                    batch["short"][i] = pyln.normalize.loudness(batch["short"][i], loudness, -23.0)

                batch_size = len(batch["short"])
                results[test_name]["pesq"] += pesq(**batch) * batch_size
                results[test_name]["si_sdr"] += si_sdr(**batch) * batch_size
                n_samples += batch_size

                if gen_dir is not None:
                    for pred_audio, path, text in zip(batch["short"].detach().cpu(), batch["mix_path"], batch["text"]):
                        stem = Path(path).stem.split("-")[0]
                        sf.write(f'{str(audio_dir / stem)}.wav', pred_audio, sr)
                        with open(f'{text_dir / stem}.txt', 'w') as file:
                            file.write(text)

                for m_name, val in results[test_name].items():
                    logger.info(f'    {m_name}: {val / n_samples}')

            for m_name in results[test_name].keys():
                results[test_name][m_name] /= n_samples

    metrics_df = pd.DataFrame()
    logger.info('\n\n\n    Final:')
    for test_name, vals in results.values():
        logger.info(f'\n\n    {test_name} results:')
        for algo_name, val in vals.items():
            logger.info(f'    {algo_name}: {np.mean(vals):.6f}')
            metrics_df.loc[test_name, algo_name] = np.mean(vals)

    logger.info(f'    Saving metrics in {out_file.split(".")[0] + ".csv"}')
    metrics_df.to_csv(out_file.split('.')[0] + '.csv')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.csv",
        type=str,
        help="File to write result metrics (.csv)",
    )
    args.add_argument(
        "-g",
        "--generate-dataset",
        default=None,
        type=str,
        help="Directory to generate dataset with denoised audio"
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=5,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio")
                        },
                    }
                ],
            }
        }

    for test_name in filter(lambda name: name.startswith("test"), config["data"].keys()):
        config["data"][test_name]["batch_size"] = args.batch_size
        config["data"][test_name]["n_jobs"] = args.jobs

    main(config, args.output, args.generate_dataset)
