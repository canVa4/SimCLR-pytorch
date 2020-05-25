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


def evaluation(checkpoints_folder, config, device):
    model = ResNetSimCLR(**config['model'])
    model.eval()
    model.load_state_dict(torch.load(os.path.join(checkpoints_folder, 'model.pth')))
    model = model.to(device)

    train_dataset = torchvision.datasets.CIFAR10(
        root='../data/CIFAR10',
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        train=False,
        download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(0.05 * num_train))
    train_idx, test_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)  # ?????sampler????????????

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
                                               num_workers=0, drop_last=True, shuffle=False)

    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], sampler=test_sampler,
                                              num_workers=0, drop_last=True)

    X_train_feature = []
    y_train = []
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        features, _ = model(batch_x)
        X_train_feature.extend(features.cpu().detach().numpy())
        y_train.extend(batch_y.cpu().detach().numpy())

    X_train_feature = np.array(X_train_feature)
    y_train = np.array(y_train)

    X_test_feature = []
    y_test = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        features, _ = model(batch_x)
        X_test_feature.extend(features.cpu().detach().numpy())
        y_test.extend(batch_y.cpu().detach().numpy())
    X_test_feature = np.array(X_test_feature)
    y_test = np.array(y_test)
    scaler = preprocessing.StandardScaler()
    print('ok')
    scaler.fit(X_train_feature)
    # print(X_test_feature.shape)
    # print(y_test.shape)
    linear_model_eval(scaler.transform(X_train_feature), y_train, scaler.transform(X_test_feature), y_test)


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
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    print("KNN feature eval")
    print("Train score:", neigh.score(X_train, y_train))
    print("Test score:", neigh.score(X_test, y_test))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    folder = '../runs/May23_16-14-04_6016361f578f'
    checkpoints_folder = os.path.join(folder, 'checkpoints')
    config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))
    evaluation(checkpoints_folder, config, device)
