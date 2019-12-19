import torch


def calculate_similarity_matrix(t1, t2):
    """
    :param t1: B x N1 x C
    :param t2: B x N2 x C
    """
    return torch.bmm(t1, t2.permute(0, 2, 1))


def calculate_similarity_vector(t1, t2):
    """
    :param t1: B x N1 x C
    :param t2: B x N2 x C
    """
    return torch.sum(t1 * t2, dim=-1)


def calculate_inv_similarity_matrix(t1, t2):
    """
    :param t1: B x N1 x C
    :param t2: B x N2 x C
    """
    sim = 2 - 2 * calculate_similarity_matrix(t1, t2)
    sim = sim.clamp(min=1e-8, max=4.0)
    sim = torch.sqrt(sim)
    return sim


def calculate_inv_similarity_vector(t1, t2):
    """
    :param t1: B x N1 x C
    :param t2: B x N2 x C
    """
    sim = 2 - 2 * calculate_similarity_vector(t1, t2)
    sim = sim.clamp(min=1e-8, max=4.0)
    sim = torch.sqrt(sim)
    return sim


def calculate_distance_matrix(t1, t2):
    """
    :param t1: B x N1 x 2
    :param t2: B x N2 x 2
    """
    t1 = t1.unsqueeze(2).float()
    t2 = t2.unsqueeze(1).float()

    dist = torch.norm(t1 - t2, p=2, dim=-1)

    return dist


def symmetric_epipolar_distance(kp1, kp2, F):
    """
    :param kp1: B x N x 3; keypoints on the first image in homogeneous coordinates
    :param kp2: B x N x 3; keypoints on the second image in homogeneous coordinates
    :param F: B x 3 x 3; Fundamental matrix connecting first and second image planes
    """
    epipolar_line1 = torch.bmm(kp1, F)  # B x N x 3
    epipolar_line2 = torch.bmm(kp2, F.permute(0, 2, 1))

    epipolar_distance = (kp2 * epipolar_line1).sum(dim=2).abs()
    norm = (1 / epipolar_line1[:, :, :2].norm(2, -1) + 1 / epipolar_line2[:, :, :2].norm(2, -1))

    return epipolar_distance * norm


def robust_symmetric_epipolar_distance(kp1, kp2, F, gamma=0.5):
    """
    :param kp1: B x N x 3; keypoints on the first image in homogeneous coordinates
    :param kp2: B x N x 3; keypoints on the second image in homogeneous coordinates
    :param F: B x 3 x 3; Fundamental matrix connecting first and second image planes
    :param gamma; float
    """
    return torch.clamp(symmetric_epipolar_distance(kp1, kp2, F), max=gamma)


def create_coordinates_grid(shape):
    """
    :param shape: (b, c, h, w)
    """
    b, _, h, w = shape

    gy, gx = torch.meshgrid([torch.arange(h), torch.arange(w)])
    gx = gx.float().unsqueeze(-1)
    gy = gy.float().unsqueeze(-1)

    grid = torch.cat((gx, gy), dim=-1)

    # Repeat grid for each batch
    grid = grid.unsqueeze(0)  # 1 x H x W x 2
    grid = grid.repeat(b, 1, 1, 1)  # B x H x W x 2

    return grid
