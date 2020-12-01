import torchaudio
import torch
from model import MelDataset, DivaAutoEncoder
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import os
import shutil


class AudioFeatureExtractor:
    def __init__(self, model_path: str, map_location=torch.device('cpu')):
        self.model = DivaAutoEncoder()
        self.model.load_state_dict(torch.load(model_path, map_location=map_location))
        self.model.eval()
        self.paths = None       # np.ndarray
        self.latent_mat = None  # np.ndarray

    def encode(self, audio_path):
        # Extract the mel spectrogram
        # this step is the same as in `extract_mel.py`
        waveform, _ = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=22050
        )(waveform)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            f_min=30,
            f_max=11000,
            n_mels=64,
            n_fft=2048,
            hop_length=2048 // 2
        )(waveform)
        mel = mel[0]
        # chop off the right-most column to make it 64x64
        mel = mel[:, :-1]
        mel = torch.reshape(mel, (1, 1, mel.size(0), mel.size(1)))

        # normalize
        mel = torch.log(mel + 1)
        maxes, _ = torch.max(mel.view(mel.size(0), -1), dim=1)
        maxes = maxes.view(maxes.size(0), 1, 1)
        mel /= maxes

        # encode it into the latent space
        latent = self.model.encode(mel)
        latent = torch.reshape(latent, (1, -1)).detach().numpy()
        print(latent)

        selected_presets = self._retrieve_presets(latent)
        wav_name = os.path.basename(audio_path)[:-len('.wav')]
        selected_preset_audio_dir = f'selected_presets/{wav_name}'
        Path(selected_preset_audio_dir).mkdir(parents=True, exist_ok=True)

        # TODO: should return the preset paths to the host in the future
        # copy the preset audio to the dist folder
        for wav_path in selected_presets:
            shutil.copy(wav_path, selected_preset_audio_dir)

    def encode_dataset(self, dataset: MelDataset):
        latent_mat = []
        self.paths = dataset.paths
        data_loader = DataLoader(dataset, 10, shuffle=False)

        for i, (inputs, labels, _) in enumerate(data_loader):
            # inputs = torch.reshape(inputs, (1, 1, inputs.size(0), inputs.size(1)))
            latent = self.model.encode(inputs).detach().numpy()
            latent.resize((latent.shape[0], latent.shape[1]*latent.shape[2]*latent.shape[3]))
            latent_mat.append(latent)

        self.latent_mat = np.concatenate(latent_mat, axis=0)

    def _retrieve_presets(self, latent: np.ndarray):
        dist = np.linalg.norm(latent - self.latent_mat, axis=1)
        min_dists_ind = list(np.argsort(dist)[:5])
        selected_presets = [self.paths[i] for i in min_dists_ind]
        return selected_presets


if __name__ == '__main__':
    feature_extractor = AudioFeatureExtractor('models/auto-encoder-20201020050003.pt') # NOTE: hard coded here
    dataset = MelDataset(flat=False)
    feature_extractor.encode_dataset(dataset)
