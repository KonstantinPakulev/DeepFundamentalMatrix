import torch


def to_homogeneous_coordinates(points):
    """
    :param points: B x N x 2
    """
    ones = torch.ones((points.shape[0], points.shape[1], 1)).type_as(points).to(points.device)
    return torch.cat((points, ones), -1)


def construct_normalize_transform(scale, mean_point):
    normalize_transform = torch.zeros((mean_point.shape[0], 3, 3)).to(mean_point.device)

    normalize_transform[:, 0, 0] = scale
    normalize_transform[:, 1, 1] = scale
    normalize_transform[:, 2, 2] = 1

    normalize_transform[:, 0, 2] = -mean_point[:, 0] * scale
    normalize_transform[:, 1, 2] = -mean_point[:, 1] * scale

    return normalize_transform


def normalize_points(points):
    """
    :param points: B x N x 3
    """
    mean_point = points.mean(1)
    diff = points - mean_point.unsqueeze(1)
    mean_dist = diff[:, :, :2].norm(2, -1).mean(1)

    scale = 1 / mean_dist

    normalize_transform = construct_normalize_transform(scale, mean_point)

    points = torch.bmm(points, normalize_transform.permute(0, 2, 1))

    return points, normalize_transform


def normalize_weighted_points(points, weights):
    """
    :param points: B x N x 3
    :param weights: B x N
    """
    weights_sum = weights.sum(-1)

    mean_point = torch.sum(weights.unsqueeze(-1) * points, 1) / weights_sum.unsqueeze(-1)
    diff = points - mean_point.unsqueeze(1)
    mean_dist = (weights * diff[:, :, :2].norm(2, -1)).sum(-1) / weights_sum

    scale = 1.4142 / mean_dist

    normalize_transform = construct_normalize_transform(scale, mean_point)

    points = torch.bmm(points, normalize_transform.permute(0, 2, 1))

    return points, normalize_transform


def transform_F_to_image_space(norm_transform1, norm_transform2, F_estimate):
    F_estimate = norm_transform1.permute(0, 2, 1).bmm(F_estimate.bmm(norm_transform2))
    F_estimate = F_estimate / F_estimate[:, -1, -1].unsqueeze(-1).unsqueeze(-1)
    return F_estimate