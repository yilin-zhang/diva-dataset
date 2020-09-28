import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import CHARACTER_TABLE


# https://necromuralist.github.io/neural_networks/posts/normalizing-with-numpy/
def normalize_cols(x: np.ndarray):
    """
    function that normalizes each col of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by col) numpy matrix.
    """
    return x/np.linalg.norm(x, ord=1, axis=0, keepdims=True)


with open('dataset/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('dataset/binary_characters.pkl', 'rb') as f:
    binary_characters = pickle.load(f)

features = normalize_cols(features)

# initialize dimension reduction models
# pca = PCA(n_components=2)
# kpca = KernelPCA(n_components=2, kernel='poly')
tsne = TSNE(n_components=2, random_state=0)

# reduce dimensions
features_tsne = tsne.fit_transform(features)


# plot
color_1 = 'tab:orange'
color_2 = 'tab:green'

for idx, pair in enumerate(CHARACTER_TABLE):
    word_1, word_2 = pair
    idx_1 = np.where(1 == binary_characters[:, 2*idx])
    idx_2 = np.where(1 == binary_characters[:, 2*idx+1])
    samples_1 = features_tsne[idx_1]
    samples_2 = features_tsne[idx_2]
    # fig, ax = plt.subplots()

    plt.subplot(6, 2, idx+1)
    plt.scatter(samples_1[:, 0], samples_1[:, 1], c=color_1, label=word_1, alpha=0.3)
    plt.scatter(samples_2[:, 0], samples_2[:, 1], c=color_2, label=word_2, alpha=0.3)
    plt.legend()

plt.show()
