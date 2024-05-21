import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import gzip
import os
import torchvision


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name,
                                              label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    return x_train, y_train


def save(model):
    save_filename = 'model.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


def data_generator(root, batch_size):
    path = root + '\old\\20class\FlowAllLayers\\';
    train_set = DealDataset(path, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    test_set = DealDataset(path, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
