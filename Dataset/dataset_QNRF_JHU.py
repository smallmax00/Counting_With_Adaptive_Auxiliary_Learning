from torch.utils import data
from PIL import Image
import numpy as np
import h5py
import os
import glob
from torchvision.transforms import functional

class Dataset(data.Dataset):
    def __init__(self, data_path, dataset, is_train):
        self.is_train = is_train
        self.dataset = dataset

        if dataset == 'QNRF':
            dataset = 'QNRF'
            if is_train:
                is_train = 'train'
            else:
                is_train = 'test'
        elif dataset == 'JHU':
            dataset = 'JHU'
            if is_train:
                is_train = 'train'
            else:
                is_train = 'test'

        self.image_list = glob.glob(os.path.join(data_path, dataset, is_train, 'img', '*.jpg'))
        self.label_list = glob.glob(os.path.join(data_path, dataset, is_train, 'new_data', '*.h5'))
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = h5py.File(self.label_list[index], 'r')
        gt = np.array(label['gt'], dtype=np.float32)
        height, width = image.size[1], image.size[0]
        height = round(height / 128) * 128
        width = round(width / 128) * 128

        image = image.resize((width, height), Image.BILINEAR)
        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, gt, self.image_list[index]

    def __len__(self):
        return len(self.image_list)


