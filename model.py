import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import trange


class MelDataset(Dataset):
    def __init__(self, flat=False):
        with open('dataset/mel.pkl', 'rb') as f:
            features = pickle.load(f)

        with open('dataset/labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        self.labels = labels

        if flat:
            self.features = torch.reshape(
                features,
                shape=(features.size(0),
                       features.size(1) * features.size(2)))
        else:
            self.features = torch.reshape(features, (features.size(0),
                                                     1,
                                                     features.size(1),
                                                     features.size(2)))

        print('features size:', self.features.size())

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.size(0)


# DNN (linear)
class MyDNN(nn.Module):
    def __init__(self):
        super(MyDNN, self).__init__()
        self.lin1 = nn.Linear(8320, 4096, bias=True)
        self.bn1 = nn.BatchNorm1d(4096)
        self.lin2 = nn.Linear(4096, 2048, bias=True)
        self.bn2 = nn.BatchNorm1d(2048)
        self.lin3 = nn.Linear(2048, 1024, bias=True)
        self.bn3 = nn.BatchNorm1d(1024)
        self.lin4 = nn.Linear(1024, 512, bias=True)
        self.bn4 = nn.BatchNorm1d(512)
        self.lin5_1 = nn.Linear(512, 3, bias=True)
        self.lin5_2 = nn.Linear(512, 3, bias=True)
        self.lin5_3 = nn.Linear(512, 3, bias=True)
        self.lin5_4 = nn.Linear(512, 3, bias=True)
        self.lin5_5 = nn.Linear(512, 3, bias=True)
        self.lin5_6 = nn.Linear(512, 3, bias=True)
        self.lin5_7 = nn.Linear(512, 3, bias=True)
        self.lin5_8 = nn.Linear(512, 3, bias=True)
        self.lin5_9 = nn.Linear(512, 3, bias=True)
        self.lin5_10 = nn.Linear(512, 3, bias=True)
        self.lin5_11 = nn.Linear(512, 3, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))
        x = F.relu(self.bn4(self.lin4(x)))
        x1 = self.softmax(self.lin5_1(x))
        x2 = self.softmax(self.lin5_2(x))
        x3 = self.softmax(self.lin5_3(x))
        x4 = self.softmax(self.lin5_4(x))
        x5 = self.softmax(self.lin5_5(x))
        x6 = self.softmax(self.lin5_6(x))
        x7 = self.softmax(self.lin5_7(x))
        x8 = self.softmax(self.lin5_8(x))
        x9 = self.softmax(self.lin5_9(x))
        x10 = self.softmax(self.lin5_10(x))
        x11 = self.softmax(self.lin5_11(x))
        return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)  # dilation is not added here
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drp = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 64, 3)
        self.pool2 = nn.MaxPool2d(4, stride=4)
        self.fc1 = nn.Linear(14*6, 32)

        self.fc2_1 = nn.Linear(512, 3, bias=True)
        self.fc2_2 = nn.Linear(512, 3, bias=True)
        self.fc2_3 = nn.Linear(512, 3, bias=True)
        self.fc2_4 = nn.Linear(512, 3, bias=True)
        self.fc2_5 = nn.Linear(512, 3, bias=True)
        self.fc2_6 = nn.Linear(512, 3, bias=True)
        self.fc2_7 = nn.Linear(512, 3, bias=True)
        self.fc2_8 = nn.Linear(512, 3, bias=True)
        self.fc2_9 = nn.Linear(512, 3, bias=True)
        self.fc2_10 = nn.Linear(512, 3, bias=True)
        self.fc2_11 = nn.Linear(512, 3, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drp(F.relu(self.conv1(x)))
        x = self.drp(F.relu(self.conv2(x)))
        x = self.drp(self.pool1(F.relu(self.conv3(x))))
        x = self.drp(F.relu(self.conv4(x)))
        x = self.drp(self.pool2(F.relu(self.conv5(x))))
        return x
        # x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))
        # x = F.relu(self.lin4(x))
        # x1 = self.softmax(self.lin5_1(x))
        # x2 = self.softmax(self.lin5_1(x))
        # x3 = self.softmax(self.lin5_1(x))
        # x4 = self.softmax(self.lin5_1(x))
        # x5 = self.softmax(self.lin5_1(x))
        # x6 = self.softmax(self.lin5_1(x))
        # x7 = self.softmax(self.lin5_1(x))
        # x8 = self.softmax(self.lin5_1(x))
        # x9 = self.softmax(self.lin5_1(x))
        # x10 = self.softmax(self.lin5_1(x))
        # x11 = self.softmax(self.lin5_1(x))
        # return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11


def train_dnn(device, train_set, val_set, num_epochs, batch_size):
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, len(val_set), shuffle=True)

    net = MyDNN()
    net.to(device)

    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('runs/test1')

    for epoch in range(num_epochs):
        print('epoch:', epoch, '...')
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            out = net(inputs)

            loss = []
            for j in range(11):
                loss.append(criterion(out[j], labels[:, j]))
            loss = sum(loss)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss, epoch)

        # validate every 5 epochs
        if (epoch != 0 and epoch % 5 == 0):
            # val_inputs, val_labels = iter(val_loader).next()
            accuracies = [[]] * 11
            # NOTE: Since I set only one batch for validation, the for loop
            # is only for one iteration
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_out = net(val_inputs)
                val_loss = []
                for j in range(11):
                    val_loss.append(criterion(val_out[j], val_labels[:, j]))
                    pred = val_out[j].argmax(dim=1)
                    corrects = (pred == val_labels[:, j])
                    accuracies[j].append(corrects.sum().float() / float(val_labels.size(0)))
                val_loss = sum(val_loss)

            writer.add_scalar('Loss/val', val_loss, epoch)
            accuracies = torch.tensor(accuracies)
            print(f'accuracy size: {accuracies.size()}')
            accuracies = torch.mean(accuracies, dim=1)
            for k in range(11):
                print(f'accuracy {k}: {accuracies[k]}')

        print('finished')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    batch_size = 50

    lengths = [2795, 350, 349]
    dataset = MelDataset(True)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, lengths)

    train_dnn(device, train_set, val_set, num_epochs, batch_size)
