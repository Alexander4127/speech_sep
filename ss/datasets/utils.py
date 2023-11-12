from copy import deepcopy
import os

import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import torch
import torchaudio


def snr_mixer(clean, noise, snr):
    amp_noise = np.linalg.norm(clean) / 10**(snr / 20)
    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise
    mix = clean + noise_norm
    return mix


def vad_merge(w, top_db):
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def cut_audios(s1, s2, sec, sr, equal_lengths: bool = False):
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)

    s1_cut = []
    s2_cut = []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])

        segment += 1

    if equal_lengths and cut_len < len1:
        assert len1 == len2
        s1_r, s2_r = s1[cut_len * len(s2_cut):], s2[cut_len * len(s1_cut):]
        s1_r = np.append(s1_r, np.zeros(len(s1) - cut_len))
        s2_r = np.append(s2_r, np.zeros(len(s2) - cut_len))
        assert len(s1_r) == len(s2_r) == cut_len
        s1_cut.append(s1_r)
        s2_cut.append(s2_r)

    return s1_cut, s2_cut


# def split_batch(batch):
#     keys = ["mix", "target"]
#     cut_mix_1, cut_mix_2 = cut_audios(batch["short"])
#     assert len(cut_mix_1) == len(cut_mix_2)
#     for idx in range(len(cut_mix_1)):
#
#         for key in keys:
#             current_batch = [batch[key] for set(batch.keys()) - set(keys)]


def stack_batch(batches, len_batch, keys=("short", "target")):
    result_batch = deepcopy(batches[0])
    for key in keys:
        result_batch[key] = torch.stack([batch[key] for batch in batches]).reshape(-1)[:len_batch]
    return result_batch


def fix_length(s1, s2, min_or_max='max'):
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2


def create_mix(idx, triplet, snr_levels, out_dir, audio_len=3, test=False, sr=16000, trim_db=20, vad_db=20):
    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    ref_path = triplet["ref"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))

    meter = pyln.Meter(sr)

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    louds_ref = meter.integrated_loudness(ref)

    s1_norm = pyln.normalize.loudness(s1, louds1, -29)
    s2_norm = pyln.normalize.loudness(s2, louds2, -29)
    ref_norm = pyln.normalize.loudness(ref, louds_ref, -23.0)

    amp_s1 = np.max(np.abs(s1_norm))
    amp_s2 = np.max(np.abs(s2_norm))
    amp_ref = np.max(np.abs(ref_norm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return

    if trim_db:
        ref, _ = librosa.effects.trim(ref_norm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1_norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2_norm, top_db=trim_db)

    if len(ref) < sr:
        return

    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")

    snr = np.random.choice(snr_levels, 1).item()

    d_paths = []
    if not test:
        s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audio_len, sr)

        for i in range(len(s1_cut)):
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

            louds1 = meter.integrated_loudness(s1_cut[i])
            s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -23.0)
            loud_mix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loud_mix, -23.0)

            path_mix_i = path_mix.replace("-mixed.wav", f"_{i}-mixed.wav")
            path_target_i = path_target.replace("-target.wav", f"_{i}-target.wav")
            path_ref_i = path_ref.replace("-ref.wav", f"_{i}-ref.wav")
            sf.write(path_mix_i, mix, sr)
            sf.write(path_target_i, s1_cut[i], sr)
            sf.write(path_ref_i, ref, sr)
            t_info = torchaudio.info(str(path_target_i))
            length = t_info.num_frames / t_info.sample_rate
            d_paths.append(
                {
                    "mix_path": path_mix_i,
                    "target_path": path_target_i,
                    "ref_path": path_ref_i,
                    "audio_len": length,
                }
            )
    else:
        s1, s2 = fix_length(s1, s2, 'max')
        mix = snr_mixer(s1, s2, snr)
        louds1 = meter.integrated_loudness(s1)
        s1 = pyln.normalize.loudness(s1, louds1, -23.0)

        loud_mix = meter.integrated_loudness(mix)
        mix = pyln.normalize.loudness(mix, loud_mix, -23.0)

        sf.write(path_mix, mix, sr)
        sf.write(path_target, s1, sr)
        sf.write(path_ref, ref, sr)

        t_info = torchaudio.info(str(path_target))
        length = t_info.num_frames / t_info.sample_rate
        d_paths.append(
            {
                "mix_path": path_mix,
                "target_path": path_target,
                "ref_path": path_ref,
                "audio_len": length,
            }
        )

    return d_paths
