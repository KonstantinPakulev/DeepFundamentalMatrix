import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from source.utils.model_utils import sample_descriptors
from source.utils.image_utils import warp_image, get_visible_keypoints_mask, \
    warp_points, select_keypoints
from source.utils.metric_utils import compute_error


def evaluate(kp1, kp2, F_estimate, F_gt, thresh=1.0):
    accuracy = 0
    f1_score = 0

    for j in range(kp1.shape[0]):
        j_a, j_f1 = compute_error(kp1[j], kp2[j], F_estimate[j], F_gt[j], thresh)

        accuracy += j_a
        f1_score += j_f1

    accuracy /= kp1.shape[0]
    f1_score /= kp1.shape[0]

    return accuracy, f1_score


"""
Batch keys
"""
IMAGE1_NAME = 'image1_name'
IMAGE2_NAME = 'image2_name'
IMAGE1 = 'image1'
IMAGE2 = 'image2'
HOMO12 = 'homo12'
HOMO21 = 'homo21'
S_IMAGE1 = 's_image1'
S_IMAGE2 = 's_image2'

KP1 = 'kp1'
KP2 = 'kp2'
KP1_DESC = 'kp1_desc'
KP2_DESC = 'kp2_desc'
W_KP1 = 'w_kp1'
W_KP2 = 'w_kp2'
WV_KP1_MASK = 'wv_kp1_mask'
WV_KP2_MASK = 'wv_kp2_mask'


def forward(model, batch, device):
    image1, image2, homo12, homo21 = (
        batch[IMAGE1].to(device),
        batch[IMAGE2].to(device),
        batch[HOMO12].to(device),
        batch[HOMO21].to(device))

    output1 = model(image1)
    output2 = model(image2)

    score1, score2 = output1[0], output2[0]
    desc1, desc2 = output1[1], output2[1]

    w_score1 = warp_image(score2.shape, score1, homo21)
    w_score2 = warp_image(score1.shape, score2, homo12)

    nms_thresh = 0
    nms_k_size = 5
    top_k = 512

    kp1 = select_keypoints(score1, nms_thresh, nms_k_size, top_k)
    kp2 = select_keypoints(score2, nms_thresh, nms_k_size, top_k)

    kp1_desc = sample_descriptors(desc1, kp1, 8)
    kp2_desc = sample_descriptors(desc2, kp2, 8)

    w_kp1 = warp_points(kp1, homo12)
    w_kp2 = warp_points(kp2, homo21)

    wv_kp1_mask = get_visible_keypoints_mask(image2, w_kp1)
    wv_kp2_mask = get_visible_keypoints_mask(image1, w_kp2)

    endpoint = {
        KP1: kp1,
        KP2: kp2,

        KP1_DESC: kp1_desc,
        KP2_DESC: kp2_desc,

        W_KP1: w_kp1,
        W_KP2: w_kp2,

        WV_KP1_MASK: wv_kp1_mask,
        WV_KP2_MASK: wv_kp2_mask,
    }

    if S_IMAGE1 in batch and S_IMAGE2 in batch:
        endpoint[S_IMAGE1] = batch[S_IMAGE1]
        endpoint[S_IMAGE2] = batch[S_IMAGE2]

    return endpoint


def torch2cv(img, normalize=False, to_rgb=False):
    """
    :param img: C x H x W
    :param normalize: normalize image by max value
    :param to_rgb: convert image to rgb from grayscale
    """
    if normalize:
        img = img / img.max()

    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def cv2torch(img):
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    return img


def to_cv2_keypoint(kp):
    """
    :param kp: N x 2
    """
    if torch.is_tensor(kp):
        kp = kp.detach().cpu().numpy()
    kp = list(map(lambda x: cv2.KeyPoint(x[1], x[0], 0), kp))

    return kp


def to_cv2_dmatch(kp, matches):
    """
    :param kp: N x 2
    :param matches: N
    """
    matches = matches.detach().cpu().numpy()
    return list(map(lambda x: cv2.DMatch(x[0], x[1], 0, 0), zip(np.arange(0, kp.shape[0]), matches)))


def draw_cv_keypoints(cv_image, kp, color):
    """
    :param cv_image: H x W x C
    :param kp: N x 2
    :param color: tuple (r, g, b)
    """
    cv_kp = to_cv2_keypoint(kp)
    return cv2.drawKeypoints(cv_image, cv_kp, None, color=color)


def draw_cv_matches(cv_image1, cv_image2, kp1, kp2, matches, match_mask):
    """
    :param cv_image1: H x W x C, numpy array
    :param cv_image2: H x W x C, numpy array
    :param kp1: N x 2, torch tensor
    :param kp2: N x 2, torch tensor
    :param matches: N, torch int tensor
    :param match_mask: N, torch bool tensor
    """
    cv_kp1 = to_cv2_keypoint(kp1)
    cv_kp2 = to_cv2_keypoint(kp2)

    matches = to_cv2_dmatch(kp1, matches)
    match_mask = match_mask.detach().cpu().byte().numpy().ravel().tolist()

    return cv2.drawMatches(cv_image1, cv_kp1, cv_image2, cv_kp2,
                           matches, None,
                           matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                           matchesMask=match_mask)


def plot_figures(figures, nrows=1, ncols=1, size=None):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axes_list = plt.subplots(ncols=ncols, nrows=nrows, figsize=size)
    for ind, title in zip(range(len(figures)), figures):
        if nrows * ncols != 1:
            axes_list.ravel()[ind].imshow(figures[title], cmap=plt.jet())
            axes_list.ravel()[ind].set_title(title)
            axes_list.ravel()[ind].set_axis_off()
        else:
            axes_list.imshow(figures[title], cmap=plt.jet())
            axes_list.set_title(title)
            axes_list.set_axis_off()

    plt.tight_layout()  # optional
