import pickle
import numpy as np
from config import CHARACTER_TABLE

##
# Check the skewness
with open('dataset/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

labels = labels.numpy()

for i in range(11):
    feature = labels[:, i]

    print(f'In feature pair {CHARACTER_TABLE[i]}')

    num_no_label = np.sum(feature == 0)
    print(f'No label: {num_no_label}')

    num_char_1 = np.sum(feature == 1)
    print(f'{CHARACTER_TABLE[i][0]}: {num_char_1}')

    num_char_2 = np.sum(feature == 2)
    print(f'{CHARACTER_TABLE[i][1]}: {num_char_2}')

