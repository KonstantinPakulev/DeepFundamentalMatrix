import os
import numpy as np
import pandas as pd
from skimage import io

from torch.utils.data import Dataset

from source.utils.eval_utils import (IMAGE1_NAME, IMAGE2_NAME,
                                     IMAGE1, IMAGE2,
                                     HOMO12, HOMO21,
                                     S_IMAGE1, S_IMAGE2)


class HPatchesDataset(Dataset):

    def __init__(self, root_path, csv_path, item_transforms=None, sources=False):
        """
        :param root_path: Path to the dataset folder
        :param csv_path: Path to csv file with annotations
        :param item_transforms: Transforms for both homography and image
        :param sources:
        """
        self.root_path = root_path
        self.annotations = pd.read_csv(csv_path)
        self.item_transforms = item_transforms
        self.sources = sources

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, id):
        folder = self.annotations.iloc[id, 0]
        image1_name = self.annotations.iloc[id, 1]
        image2_name = self.annotations.iloc[id, 2]

        image1_path = os.path.join(self.root_path, folder, image1_name)
        image2_path = os.path.join(self.root_path, folder, image2_name)

        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)

        homo12 = np.asmatrix(self.annotations.iloc[id, 3:].values).astype(np.float).reshape(3, 3)
        homo21 = homo12.I

        item = {IMAGE1_NAME: folder + "_" + image1_name,
                IMAGE2_NAME: folder + "_" + image2_name,
                IMAGE1: image1, IMAGE2: image2,
                HOMO12: homo12, HOMO21: homo21}

        if self.sources:
            item[S_IMAGE1] = image1.copy()
            item[S_IMAGE2] = image2.copy()

        if self.item_transforms:
            item = self.item_transforms(item)

        return item