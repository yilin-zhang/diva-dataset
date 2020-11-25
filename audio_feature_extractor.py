import torchaudio
import torch
from model import DivaAutoEncoder

class AudioFeatureExtractor:
    def __init__(self, model_path: str, map_location=torch.device('cpu')):
        self.model = DivaAutoEncoder()
        self.model.load_state_dict(torch.load(model_path, map_location=map_location))
        self.model.eval()

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

        # encode it into the latent space
        latent = self.model.encode(mel)
        latent = torch.reshape(latent, (1, -1))
        print(latent)



