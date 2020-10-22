import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import MelDataset, DivaAutoEncoder
from config import CHARACTER_TABLE

torch.manual_seed(0)
np.random.seed(0)
dataset = MelDataset(flat=False, normalize=True)

# lengths = [2795, 350, 349]
#_, _, dataset = torch.utils.data.random_split(dataset, lengths)

ae_model = DivaAutoEncoder()
ae_model.load_state_dict(torch.load('models/auto-encoder-20201020050003.pt', map_location=torch.device('cpu')))
ae_model.eval()


### Test latent space
def get_latent(device, net, features: torch.tensor) -> np.ndarray:
    batch_size = 8
    data_loader = DataLoader(features, batch_size, shuffle=False)
    net.to(device)
    latent_batches = []
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        latent = net.encode(inputs)
        latent_batches.append(latent)
    latent = torch.cat(latent_batches, 0)
    latent = latent.view(latent.size(0), -1).detach().numpy()
    return latent


def tsne(features):
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)

    return features_tsne


def plot_tsne(features_tsne, labels, character_id):
    Path('plots/latent').mkdir(parents=True, exist_ok=True)
    color_1 = 'tab:orange'
    color_2 = 'tab:green'

    word_1, word_2 = CHARACTER_TABLE[character_id]
    idx_1 = np.where(1 == labels[:, character_id])
    idx_2 = np.where(2 == labels[:, character_id])

    samples_1 = features_tsne[idx_1]
    samples_2 = features_tsne[idx_2]

    plt.clf()
    plt.scatter(samples_1[:, 0], samples_1[:, 1], c=color_1, label=word_1, alpha=0.3)
    plt.scatter(samples_2[:, 0], samples_2[:, 1], c=color_2, label=word_2, alpha=0.3)
    plt.legend()
    plt.savefig('plots/latent/' + str(character_id) + '-' + word_1 + '-' + word_2)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
latent = get_latent(device, ae_model, dataset)
latent_tsne = tsne(latent)
labels_np = dataset.labels.numpy()
for i in range(11):
    plot_tsne(latent_tsne, labels_np, i)

### Test forward
# feature, _ = dataset[3]
# feature_np = feature.view(feature.size(1), -1).numpy()
# ae_feature = ae_model.forward(feature.view(1, *feature.size()))
# print(f'ae_feature size: {ae_feature.size()}')
# ae_feature_np = ae_feature.view(ae_feature.size(2), ae_feature.size(3)).detach().numpy()
#
# plt.figure()
# plt.imshow(feature_np)
# plt.figure()
# plt.imshow(ae_feature_np)
# plt.show()

### Test encode
# encode_feature = ae_model.encode(feature.view(1, *feature.size()))
# print(f'encode_feature size: {encode_feature.size()}')

# print(feature.size())
