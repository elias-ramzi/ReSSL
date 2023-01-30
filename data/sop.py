from typing import Tuple, List, Callable, Optional, Union
from os.path import join, expanduser, expandvars

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision

NoneType = type(None)


def set_labels_to_range(labels: np.ndarray) -> np.ndarray:
    """
    set the labels so it follows a range per level of semantic
    """
    new_labels = []
    for lvl in range(labels.shape[1]):
        unique = sorted(set(labels[:, lvl]))
        conversion = {x: i for i, x in enumerate(unique)}
        new_lvl_labels = [conversion[x] for x in labels[:, lvl]]
        new_labels.append(new_lvl_labels)

    return np.stack(new_labels, axis=1)


class SOPContrastive(Dataset):

    def __init__(
        self,
        data_dir: str = '/gpfsscratch/rech/nfj/uef86cm/Stanford_Online_Products',
        mode: str = 'train',
        aug: Optional[Callable] = None,
    ) -> NoneType:
        super().__init__()
        self.data_dir = expandvars(expanduser(data_dir))
        self.mode = mode

        if mode == 'train':
            mode = ['train']
            self.transform = aug
        elif mode == 'test':
            mode = ['test']
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=[256, 256]),
                torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        elif mode == 'all':
            mode = ['train', 'test']
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=[256, 256]),
                torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        labels = []
        super_labels = []
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir, f'Ebay_{splt}.txt'), sep=' ')
            self.paths.extend(gt["path"].apply(lambda x: join(self.data_dir, x)).tolist())
            labels.extend((gt["class_id"] - 1).tolist())
            super_labels.extend((gt["super_class_id"] - 1).tolist())

        self.labels = np.stack([labels, super_labels], axis=1)
        self.labels = set_labels_to_range(self.labels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        img = Image.open(self.paths[idx]).convert('RGB')
        if isinstance(self.transform, list):
            images = [t(img) for t in self.transform]
        elif self.transform:
            images = self.transform(img)

        return images
