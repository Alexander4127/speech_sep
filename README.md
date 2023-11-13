# Speech separation project

## Report

The introduction, technical details, and results are presented in the [wandb report](https://wandb.ai/practice-cifar/ss_project/reports/Speech-Separation-Report--Vmlldzo1OTUyNzgx).

## Installation guide

To get started install the requirements
```shell
pip install -r ./requirements.txt
```

## Model training

[SpEX+](https://www.isca-speech.org/archive/pdfs/interspeech_2020/ge20_interspeech.pdf) architecture with 
additional speaker classification head was implemented in this project.

To train model from scratch run
```shell
python3 train.py -c final_model/config.json
```

For fine-tuning pretrained model from checkpoint, `--resume` parameter is applied. For example, fine-tuning pretrained model
organized as follows
```shell
python3 train.py -c final_model/finetune.json -r saved/models/pretrain_final/<run_id>/model_best.pth
```

This command generates new mixed dataset. This option can be disabled by passing `"reuse": true` for 
train dataset in config `final_model/finetune.json`.

## Model applications

Before applying model pretrained checkpoint is loaded by python code
```python3
import gdown
gdown.download("https://drive.google.com/uc?id=19i4NIk8R8AlkGvMfhQl8ex-eCg4g2Isv", "default_test_model/checkpoint.pth")
```

Model evaluation is executed by command
```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.csv \
   -g <output_dir> \
   -s <interval_len>
```

Where `-o` specify output `.csv` file, which represents metrics
- PESQ (Perceptual Evaluation of Speech Quality)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

Further sections reveal other command line arguments.

### Inference results

*Important remark*: for further experiments `test-clean` part of Librispeech dataset is required.
It will be automatically installed after at least one execution `python3 test.py` with default arguments.

Model evaluation with custom data conducted by running `test.py`
```shell
python3 test.py -t path/to/custom/dir
```

This command executes model on custom dataset folder, which includes `mix`, `target` and `ref` subdirectories 
with filenames `*-mixed.wav`, `*-target.wav` and `*-ref.wav` respectively. Such directory will be created in
`data/datasets/mixed/data/custom` from mixed `test-clean` dataset after running
```shell
bash custom_set.sh
```

### Generating dataset for automatic speech recognition

Extracted audio for the test set can be gathered in one directory by executing
```shell
python3 test.py -g path/to/output/dir
```

This results can be compared with direct speech recognition from mixed audio.
Speech recognition pipeline was taken from [asr repository](https://github.com/Alexander4127/asr/).
Comparison of mixed and extracted audios quality carried out by
```shell
bash asr_score.sh
```

### Audio segmentation on the inference stage

For training stability audio were split into 3-seconds interval. However, test data provides
arbitrary lengths of audios, which can be divided into intervals on inference stage with
```shell
python3 test.pt -s <interval_len_in_seconds>
```

### Noised audio with WHAM!

[WHAM!](http://wham.whisper.ai/) dataset provides diverse background noise, which can be also mixed with
input audio. Installation
```shell
wget https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
unzip wham_noise.zip
```

Creating noised dataset and model evaluation
```shell
python3 test.py -c wham_test/config.json
```

Described pipeline only involves `tt (test)` part of `WHAM!`, therefore, other directories are not required. 

## Credits

This repository is based on an [asr-template](https://github.com/WrathOfGrapes/asr_project_template) repository.
