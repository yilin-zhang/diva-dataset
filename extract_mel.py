import numpy as np
import torch
import torchaudio
import os
import csv
import pickle
from pathlib import Path


def transform_diva_labels(label_array):
    ''' Transform a list of labels to one-hot encoding
    Arg:
    - label_list: an array of labels
    Return:
    - on-hot encoding

    00 -> 0
    10 -> 1
    01 -> 2
    '''
    encoded_list = []
    for i in range(np.size(label_array)//2):
        bit_1 = label_array[2*i]
        bit_2 = label_array[2*i+1]
        if bit_1 == 0 and bit_2 == 0:
            encoded_list.append(0)
        elif bit_1 == 1 and bit_2 == 0:
            encoded_list.append(1)
        elif bit_1 == 0 and bit_2 == 1:
            encoded_list.append(2)

    return np.array(encoded_list)


def transform_back_diva_labels(encoded_array):
    '''
    0 -> 00
    1 -> 10
    2 -> 01
    '''
    label_list = []
    for i in range(np.size(encoded_array)):
        bit = encoded_array[i]
        if bit == 0:
            label_list += [0, 0]
        elif bit == 1:
            label_list += [1, 0]
        elif bit == 2:
            label_list += [0, 1]

    return np.array(label_list)


def encode_dataset(dataset_path):
    csv_path = os.path.join(dataset_path, 'meta.csv')

    mel_list = []
    encoded_label_list = []
    with open(csv_path, 'r') as csvfile:
        meta_reader = csv.reader(csvfile)
        next(meta_reader)
        i = 1
        for row in meta_reader:
            labels = np.array(row[:-1], dtype=np.uint8)
            audio_path = os.path.join(dataset_path, row[-1])
            print(audio_path)

            # process labels
            encoded_label_list.append(transform_diva_labels(labels))

            # process the waveform
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
                hop_length=2048//2
            )(waveform)
            mel = mel[0].numpy()
            # chop off the right-most column to make it 64x64
            mel = mel[:, :-1]
            mel_list.append(mel)

            print("Iteration:", i)
            i += 1

    return torch.tensor(mel_list), torch.tensor(encoded_label_list)


if __name__ == '__main__':
    DATASET_PATH = 'diva-preset-audio-dataset'
    EXPORT_DIR = 'dataset'
    mel, labels = encode_dataset(DATASET_PATH)

    # Dump the data
    print('size of mel matrix:', mel.shape)
    print('size of labels matrix:', labels.shape)

    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    # Dump the data
    with open(os.path.join(EXPORT_DIR, 'mel.pkl'), 'wb') as f:
        pickle.dump(mel, f)

    with open(os.path.join(EXPORT_DIR, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)
