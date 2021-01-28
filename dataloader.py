import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils import unpickle

class Cifar10Dataset(Dataset):

    def __init__(self,
                 basepath : str = "cifar-10-batches-py",
                 train : bool = True,
                 transform=None
                 ) -> None:

        self.basepath = basepath
        self.transform = transform
        self.train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                                'data_batch_4', 'data_batch_5',]
        self.test_list = ["test_batch"]

        self.data = []
        self.labels = []

        if train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        for filename in downloaded_list:
            filepath = os.path.join(self.basepath, filename)
            dict = unpickle(filepath)
            self.data.append(dict[b'data'])
            self.labels.extend(dict[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx : int):
        image, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    dataloader = Cifar10Dataset()
    image, label = dataloader[0]
    print(image.shape, label)