import numpy as np

import torch
import torchvision.transforms.functional as F

from source.utils.eval_utils import (IMAGE1, IMAGE2,
                                     HOMO12, HOMO21,
                                     S_IMAGE1, S_IMAGE2)
from source.utils.image_utils import resize_homography


class ToPILImage(object):

    def __call__(self, item):
        item[IMAGE1] = F.to_pil_image(item[IMAGE1])
        item[IMAGE2] = F.to_pil_image(item[IMAGE2])

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.to_pil_image(item[S_IMAGE1])
            item[S_IMAGE2] = F.to_pil_image(item[S_IMAGE2])

        return item


class GrayScale(object):

    def __call__(self, item):
        item[IMAGE1] = F.to_grayscale(item[IMAGE1])
        item[IMAGE2] = F.to_grayscale(item[IMAGE2])

        return item


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, item):
        ratio1 = (self.size[1] / item[IMAGE1].size[0], self.size[0] / item[IMAGE1].size[1])
        ratio2 = (self.size[1] / item[IMAGE2].size[0], self.size[0] / item[IMAGE2].size[1])

        item[HOMO12] = resize_homography(item[HOMO12], ratio1, ratio2)
        item[HOMO21] = resize_homography(item[HOMO21], ratio2, ratio1)

        item[IMAGE1] = F.resize(item[IMAGE1], self.size)
        item[IMAGE2] = F.resize(item[IMAGE2], self.size)

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.resize(item[S_IMAGE1], self.size)
            item[S_IMAGE2] = F.resize(item[S_IMAGE2], self.size)

        return item


class ToTensor(object):

    def __call__(self, item):
        item[IMAGE1] = F.to_tensor(item[IMAGE1])
        item[IMAGE2] = F.to_tensor(item[IMAGE2])

        item[HOMO12] = torch.from_numpy(np.asarray(item[HOMO12])).float()
        item[HOMO21] = torch.from_numpy(np.asarray(item[HOMO21])).float()

        if S_IMAGE1 in item:
            item[S_IMAGE1] = F.to_tensor(item[S_IMAGE1])
            item[S_IMAGE2] = F.to_tensor(item[S_IMAGE2])

        return item