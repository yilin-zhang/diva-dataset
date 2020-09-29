import pickle
import numpy as np
from pathlib import Path
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

def zscore(x: np.ndarray):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    return (x-mean)/std

def load_features():
    with open('dataset/features.pkl', 'rb') as f:
        features = pickle.load(f)

    with open('dataset/binary_characters.pkl', 'rb') as f:
        binary_characters = pickle.load(f)

    return features, binary_characters


def remove_strange_rows(features, binary_characters, limit=10):
    num_features = features.shape[1]
    tmp_comb_matrix = np.concatenate((features, binary_characters), axis=1)
    tmp_comb_matrix = tmp_comb_matrix[~np.any(features > limit, axis=1)]
    features = tmp_comb_matrix[:, :num_features]
    binary_characters = tmp_comb_matrix[:, num_features:]

    return features, binary_characters

def tsne(features):
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)

    return features_tsne

# plot
def plot_tsne(features_tsne, binary_characters):
    color_1 = 'tab:orange'
    color_2 = 'tab:green'

    Path('plots/tsne').mkdir(parents=True, exist_ok=True)
    for idx, pair in enumerate(CHARACTER_TABLE):
        word_1, word_2 = pair
        idx_1 = np.where(1 == binary_characters[:, 2*idx])
        idx_2 = np.where(1 == binary_characters[:, 2*idx+1])

        samples_1 = features_tsne[idx_1]
        samples_2 = features_tsne[idx_2]

        plt.clf()
        # plt.subplot(6, 2, idx+1)
        plt.scatter(samples_1[:, 0], samples_1[:, 1], c=color_1, label=word_1, alpha=0.3)
        plt.scatter(samples_2[:, 0], samples_2[:, 1], c=color_2, label=word_2, alpha=0.3)
        plt.legend()

        plt.savefig('plots/tsne/' + str(idx))


def plot_dual_features(features, binary_characters):
    feature_list_1 = (
        'spec_centroid_mean',
        'spec_crest_mean',
        'spec_decrease_mean',
        'spec_energy_mean',
        'spec_flatness_mean',
        'spec_flux_mean',
        'spec_hfc_mean',
        'spec_rolloff_mean',
        'spec_strongpeak_mean',
        'frame_zerocrossing_mean',
        'logattacktime'
    )

    feature_list_2 = (
        'spec_centroid_std',
        'spec_crest_std',
        'spec_decrease_std',
        'spec_energy_std',
        'spec_flatness_std',
        'spec_flux_std',
        'spec_hfc_std',
        'spec_rolloff_std',
        'spec_strongpeak_std',
        'frame_zerocrossing_std',
    )

    permutations_1 = [
        (x, y) for x in range(len(feature_list_1))
        for y in range(x)
    ]

    permutations_2 = [
        (x, y) for x in range(len(feature_list_2))
        for y in range(x)
    ]

    color_1 = 'tab:orange'
    color_2 = 'tab:green'

    Path('plots/dual').mkdir(parents=True, exist_ok=True)

    for idx, character in enumerate(CHARACTER_TABLE):
        character_1, character_2 = character
        for permutation in permutations_1:
            feature_1, feature_2 = permutation

            idx_1 = np.where(1 == binary_characters[:, 2*idx])
            idx_2 = np.where(1 == binary_characters[:, 2*idx+1])

            sel_features = features[:, [feature_1, feature_2]]

            samples_1 = sel_features[idx_1[0], :]
            samples_2 = sel_features[idx_2[0], :]

            plt.clf()
            plt.scatter(samples_1[:, 0], samples_1[:, 1],
                        c=color_1, label=character_1, alpha=0.3)
            plt.scatter(samples_2[:, 0], samples_2[:, 1],
                        c=color_2, label=character_2, alpha=0.3)

            plt.xlabel(feature_list_1[feature_1])
            plt.ylabel(feature_list_1[feature_2])
            plt.legend()
            plt.savefig('plots/dual/' + 'char-' + str(idx) + '-' +
                        feature_list_1[feature_1] + '-' + feature_list_1[feature_2])

        for permutation in permutations_2:
            feature_1, feature_2 = permutation

            idx_1 = np.where(1 == binary_characters[:, 2*idx])
            idx_2 = np.where(1 == binary_characters[:, 2*idx+1])

            sel_features = features[:, [len(feature_list_1) + feature_1,
                                        len(feature_list_1) + feature_2]]

            samples_1 = sel_features[idx_1[0], :]
            samples_2 = sel_features[idx_2[0], :]

            plt.clf()
            plt.scatter(samples_1[:, 0], samples_1[:, 1],
                        c=color_1, label=character_1, alpha=0.3)
            plt.scatter(samples_2[:, 0], samples_2[:, 1],
                        c=color_2, label=character_2, alpha=0.3)
            plt.xlabel(feature_list_2[feature_1])
            plt.ylabel(feature_list_2[feature_2])
            plt.legend()
            plt.savefig('plots/dual/' + 'char-' + str(idx) + '-' +
                        feature_list_2[feature_1] + '-' + feature_list_2[feature_2])


def plot_feature_matrix(features):
    plt.matshow(features)
    plt.show()


if __name__ == '__main__':
    # load features
    features, binary_characters = load_features()

    # preprocess
    features = zscore(features)
    features, binary_characters = remove_strange_rows(features, binary_characters, 5)

    # plot the result of tsne dim reduction
    features_tsne = tsne(features)
    plot_tsne(features_tsne, binary_characters)

    # print(features[2100:2210, :])
    # plot_feature_matrix(features[2100:2210, :])

    # plot feature pairs
    # plot_dual_features(features, binary_characters)
