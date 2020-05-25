import torch
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from models.resnet_simclr import ResNetSimCLR
import torchvision
import torchvision.transforms as transforms
import time

def evaluation(checkpoints_folder, config, device):
    model = ResNetSimCLR(**config['model'])
    model.eval()
    model.load_state_dict(torch.load(os.path.join(checkpoints_folder, 'model.pth')))
    model = model.to(device)

    train_set = torchvision.datasets.CIFAR10(
        root='../data/CIFAR10',
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        train=True,
        download=True)
    test_set = torchvision.datasets.CIFAR10(
        root='../data/CIFAR10',
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        train=False,
        download=True)
    # num_train = len(train_dataset)
    # indices = list(range(num_train))
    # np.random.shuffle(indices)
    #
    # split = int(np.floor(0.05 * num_train))
    # train_idx, test_idx = indices[split:], indices[:split]
    #
    # # define samplers for obtaining training and validation batches
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    # test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)  # ?????sampler????????????

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], drop_last=True, shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], drop_last=True, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=48, drop_last=True, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=48, drop_last=True, shuffle=True)


    X_train_feature = []
    label_train = []

    for data in train_loader:
        x, y = data
        x = x.to(device)
        features, _ = model(x)
        X_train_feature.extend(features.cpu().detach().numpy())
        label_train.extend(y.cpu().detach().numpy())

    X_train_feature = np.array(X_train_feature)
    label_train = np.array(label_train)

    X_test_feature = []
    label_test = []
    for data in test_loader:
        x, y = data
        x = x.to(device)
        features, _ = model(x)
        X_test_feature.extend(features.cpu().detach().numpy())
        label_test.extend(y.cpu().detach().numpy())
    X_test_feature = np.array(X_test_feature)
    label_test = np.array(label_test)
    scaler = preprocessing.StandardScaler()
    print('ok')
    scaler.fit(X_train_feature)
    # print(X_test_feature.shape)
    # print(y_test.shape)
    linear_model_eval(scaler.transform(X_train_feature), label_train, scaler.transform(X_test_feature), label_test)


def linear_model_eval(X_train, y_train, X_test, y_test):
    """
    same as the origin
    """
    clf = LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)
    print("Logistic Regression feature eval")
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))
    print("-------------------------------")
    # consider KNN to slow on my computer, so remove it
    # neigh = KNeighborsClassifier(n_neighbors=10)
    # neigh.fit(X_train, y_train)
    # print("KNN feature eval")
    # print("Train score:", neigh.score(X_train, y_train))
    # print("Test score:", neigh.score(X_test, y_test))


if __name__ == "__main__":
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    folder_name = '../runs/May25_16-01-41_6016361f578f'
    checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))
    evaluation(checkpoints_folder, config, device)
    end_time = time.time()
    print('time cost:')
    print(start_time - end_time)
