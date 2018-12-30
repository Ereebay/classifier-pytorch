import torch.utils.data as data
import os
from PIL import Image
from utils import *
from torchvision.transforms import transforms
import torch

class Flower(data.Dataset):
    def __init__(self, img_root, file_class, img_transform=None):
        super(Flower, self).__init__()
        self.img_transform = img_transform
        if self.img_transform == None:
            self.img_transform = transforms.ToTensor()
        self.data = self._load_dataset(img_root, file_class)

    def _load_dataset(self, img_root, file_class):
        output = []
        for i in file_class:
            img_dir = img_root + i['img']
            output.append(({
                'img': img_dir,
                'label': i['label']
            }))

        return output
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        img = Image.open(datum['img'])
        img = self.img_transform(img)
        label = datum['label']
        return img, label


if __name__ == '__main__':
    fileclass = read_pickle_file('./train.pickle')
    data = Flower(img_root='/Users/eree/Desktop/data/', file_class=fileclass)
    print(len(data))
    test = data.__getitem__(1)
    print(test)


