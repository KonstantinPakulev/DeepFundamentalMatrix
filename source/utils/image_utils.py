import torch
import torch.nn.functional as F

import numpy as np

from source.utils.math_utils import create_coordinates_grid


def resize_homography(homography, ratio1=None, ratio2=None):
    """
    :param homography: 3 x 3
    :param ratio1: (new_w / w, new_h / h) of the first image
    :param ratio2: (new_w / w, new_h / h) of the second image
    """
    if ratio1 is not None:
        wr1, hr1 = ratio1
        t = np.mat([[1 / wr1, 0, 0],
                    [0, 1 / hr1, 0],
                    [0, 0, 1]], dtype=homography.dtype)

        homography = homography * t

    if ratio2 is not None:
        wr2, hr2 = ratio2
        t = np.mat([[wr2, 0, 0],
                    [0, hr2, 0],
                    [0, 0, 1]], dtype=homography.dtype)

        homography = t * homography

    return homography


def warp_coordinates_grid(grid, homography):
    """
    :param grid: N x H x W x 2
    :param homography: N x 3 x 3
    """
    n, h, w, _ = grid.size()

    # Convert grid to homogeneous coordinates
    ones = torch.ones((n, h, w, 1)).type_as(grid).to(grid.device)
    grid = torch.cat((grid, ones), dim=-1)  # N x H x W x 3

    # Flatten spatial dimensions
    grid = grid.view(n, -1, 3)  # B x H * W x 3
    grid = grid.permute(0, 2, 1)  # B x 3 x H * W

    # B x 3 x 3 matmul B x 3 x H*W => B x 3 x H * W
    w_grid = torch.matmul(homography, grid)
    w_grid = w_grid.permute(0, 2, 1)  # B x H * W x 3

    # Convert coordinates from homogeneous to cartesian
    w_grid = w_grid / (w_grid[:, :, 2].unsqueeze(-1) + 1e-8)  # B x H * W x 3
    w_grid = w_grid.view(n, h, w, -1)[:, :, :, :2]  # B x H x W x 2

    return w_grid


def warp_points(points, homography):
    """
    :param points: B x N x 2, coordinates order is (h, w)
    :param homography: B x 3 x 3
    :return B x N x 2
    """
    b, n, _ = points.size()

    #  Because warping operates on x,y coordinates we also need to swap h and w
    h_keypoints = points[:, :, [1, 0]].float()
    h_keypoints = torch.cat((h_keypoints, torch.ones((b, n, 1)).to(points.device)), dim=-1)
    h_keypoints = h_keypoints.view((b, -1, 3)).permute(0, 2, 1)  # B x 3 x N

    # Warp points
    w_keypoints = torch.bmm(homography, h_keypoints)
    w_keypoints = w_keypoints.permute(0, 2, 1)  # B x N x 3

    # Convert coordinates from homogeneous to cartesian
    w_keypoints = w_keypoints / (w_keypoints[:, :, 2].unsqueeze(dim=-1) + 1e-8)
    # Restore original ordering
    w_keypoints = w_keypoints[:, :, [1, 0]].view((b, n, 2))

    return w_keypoints


def warp_image(out_shape, in_image, homo):
    """
    :param out_shape, (b, c, oH, oW)
    :param in_image: B x C x iH x iW
    :param homo: N x 3 x 3; A homography to warp coordinates from out to in
    :return w_image: B x C x H x W
    """
    _, _, h, w = out_shape

    grid = create_coordinates_grid(out_shape).to(homo.device)
    w_grid = warp_coordinates_grid(grid, homo)

    # Normalize coordinates in range [-1, 1]
    w_grid[:, :, :, 0] = w_grid[:, :, :, 0] / (w - 1) * 2 - 1
    w_grid[:, :, :, 1] = w_grid[:, :, :, 1] / (h - 1) * 2 - 1

    w_image = F.grid_sample(in_image, w_grid)  # N x C x H x W

    return w_image


def get_visible_keypoints_mask(image1, w_kp2):
    """
    :param image1: B x 1 x H x W
    :param w_kp2: B x N x 2
    :return B x N
    """
    return w_kp2[:, :, 0].ge(0) * \
           w_kp2[:, :, 0].lt(image1.size(2)) * \
           w_kp2[:, :, 1].ge(0) * \
           w_kp2[:, :, 1].lt(image1.size(3))


"""
Score processing functions
"""


def nms(score, thresh: float, k_size):
    """
    :param score: B x C x H x W
    :param thresh: float
    :param k_size: int
    :return B x C x H x w
    """
    _, _, h, w = score.size()

    score = torch.where(score < thresh, torch.zeros_like(score), score)

    pad_size = k_size // 2
    ps2 = pad_size * 2
    pad = [ps2, ps2, ps2, ps2, 0, 0]

    padded_score = F.pad(score, pad)

    slice_map = torch.tensor([], dtype=padded_score.dtype, device=padded_score.device)
    for i in range(k_size):
        for j in range(k_size):
            _slice = padded_score[:, :, i: h + ps2 + i, j: w + ps2 + j]
            slice_map = torch.cat((slice_map, _slice), 1)

    max_slice, _ = slice_map.max(dim=1, keepdim=True)
    center_map = slice_map[:, slice_map.size(1) // 2, :, :].unsqueeze(1)

    nms_mask = torch.ge(center_map, max_slice)
    nms_mask = nms_mask[:, :, pad_size: h + pad_size, pad_size: w + pad_size].type_as(score)

    score = score * nms_mask

    return score


def flat2grid(ids, w):
    """
    :param ids: B x C x N tensor of indices taken from flattened tensor of shape B x C x H x W
    :param w: Last dimension (W) of tensor from which indices were taken
    :return: B x C x N x 2 tensor of coordinates in input tensor B x C x H x W
    """
    o_h = ids // w
    o_w = ids - o_h * w

    o_h = o_h.unsqueeze(-1)
    o_w = o_w.unsqueeze(-1)

    return torch.cat((o_h, o_w), dim=-1)


def select_keypoints(score, nms_thresh, k_size, top_k):
    """
    :param score: B x 1 x H x W
    :param nms_thresh: float
    :param k_size: int
    :param top_k: int
    :return B x 1 x H x W, B x N x 2
    """
    n, c, h, w = score.size()

    # Apply nms
    score = nms(score, nms_thresh, k_size)

    # Extract maximum activation indices and convert them to keypoints
    score = score.view(n, c, -1)
    _, flat_ids = torch.topk(score, top_k)
    keypoints = flat2grid(flat_ids, w).squeeze(1)

    return keypoints