#!/bin/bash

printf "Preparing denoised dataset...\n"

mkdir denoised
python3 test.py --generate-dataset denoised
mv denoised/test/text denoised/test/transcriptions
git clone https://github.com/Alexander4127/asr
mv denoised/test asr/denoised
mkdir -p asr/noised/audio

printf "Copying .wav files to asr project\n"
mkdir tmp
cp data/datasets/mixed/data/test-clean/*-mixed.wav tmp || exit 1
for f in tmp/*-mixed.wav; do
    mv -- "$f" "${f%-mixed.wav}.wav"
done
mv tmp/* asr/noised/audio
rm -r tmp

cd asr || exit 1
cp -r denoised/transcriptions noised

printf "Installing reqs and loading model...\n"
pip install -r requirements.txt
printf "import gdown\n\n" > loader.py
echo "gdown.download('https://drive.google.com/uc?id=1kcVtCbofoos7JfTzazdXGYyPVibngjPk', 'default_test_model/checkpoint.pth')" >> loader.py
python3 loader.py
rm loader.py

printf "Running ASR with noised data...\n"
python3 test.py -t noised -o noised.json

printf "Running ASR with denoised data...\n"
python3 test.py -t denoised -o denoised.json

cd ..
mkdir asr_result
mv asr/noised.* asr_result
mv asr/denoised.* asr_result

rm -rf asr
