import pickle
import numpy as np
import csv
import os
from config import CHARACTER_TABLE, DATASET_PATH
from typing import List, Tuple
from pathlib import Path
import itertools
import os
import shutil


def check_skewness(labels: np.ndarray):
    for i in range(11):
        feature = labels[:, i]

        print(f'In feature pair {CHARACTER_TABLE[i]}')

        num_no_label = np.sum(feature == 0)
        print(f'No label: {num_no_label}')

        num_char_1 = np.sum(feature == 1)
        print(f'{CHARACTER_TABLE[i][0]}: {num_char_1}')

        num_char_2 = np.sum(feature == 2)
        print(f'{CHARACTER_TABLE[i][1]}: {num_char_2}')


def pick_data_for_validation(labels: np.ndarray) -> Tuple[List[List[int]], List[List[str]]]:
    # randomly sample 3 presets for each tag
    RATIO = 0.01  # ratio of samples that need to be picked
    characters = list(itertools.chain(*CHARACTER_TABLE))

    index_list = []
    for i in range(11):
        feature = labels[:, i].ravel()
        indices_1 = np.where(feature == 1)[0]
        indices_2 = np.where(feature == 2)[0]
        num_samples_1 = round(len(indices_1) * RATIO)
        num_samples_2 = round(len(indices_2) * RATIO)

        # make sure at least 3 samples get selected for each label
        if num_samples_1 < 3:
            num_samples_1 = 3
        if num_samples_2 < 3:
            num_samples_2 = 3

        picked_indices_1 = np.random.choice(indices_1, num_samples_1, replace=False).tolist()
        picked_indices_2 = np.random.choice(indices_2, num_samples_2, replace=False).tolist()
        index_list.append(picked_indices_1)
        index_list.append(picked_indices_2)
        print(f'{num_samples_1} samples for {characters[i*2]}')
        print(f'{num_samples_2} samples for {characters[i*2+1]}')

    # get file path list
    path_list = []
    csv_path = os.path.join(DATASET_PATH, 'meta.csv')
    with open(csv_path, 'r') as csvfile:
        meta_reader = csv.reader(csvfile)
        next(meta_reader)  # skip the first row
        meta_list = list(meta_reader)
        for label in index_list:
            sub_list = []
            for idx in label:
                sub_list.append(meta_list[idx][-1])
            path_list.append(sub_list)

    # create a directory and put audio files in it
    dirname = 'diva-subset-for-validation'
    for i, character in enumerate(characters):
        character_path = os.path.join(dirname, character)
        Path(character_path).mkdir(parents=True, exist_ok=True)
        for j, audio_path in enumerate(path_list[i]):
            # filename = os.path.basename(audio_path)
            shutil.copy(os.path.join(DATASET_PATH, audio_path), os.path.join(character_path, f'{index_list[i][j]}.wav'))

    with open(os.path.join(dirname, 'meta.csv'), 'w') as csvfile:
        meta_writer = csv.writer(csvfile)
        meta_writer.writerow(['index', 'label', 'path'])
        for i, character in enumerate(characters):
            for j, path in enumerate(path_list[i]):
                meta_writer.writerow([index_list[i][j], character, path])

    return index_list, path_list


if __name__ == '__main__':
    np.random.seed(0)

    with open('dataset/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    labels = labels.numpy()

    #check_skewness(labels)
    pick_data_for_validation(labels)
    # print(f'index_list: {index_list}')
    # print(f'path_list: {path_list}')
