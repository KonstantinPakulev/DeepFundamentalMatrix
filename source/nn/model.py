import torch
import torch.nn as nn
from torch.nn.functional import normalize

from source.utils.transform_utils import to_homogeneous_coordinates, normalize_points, normalize_weighted_points
from source.utils.math_utils import robust_symmetric_epipolar_distance

from source.utils.model_utils import (make_vgg_ms_block,
                                      make_vgg_ms_detector,
                                      make_vgg_ms_descriptor,
                                      multi_scale_nms,
                                      multi_scale_softmax)


class ModelEstimator(nn.Module):

    def __init__(self):
        super(ModelEstimator, self).__init__()

    def forward(self, kp1, kp2, weights):
        """
        :param kp1: B x N x 3
        :param kp2: B x N x 3
        :param weights: B x N
        """
        # Implementation of Hartley Eight-Point algorithm

        # Normalization of points with taking their weights into account
        norm_kp1, norm_transform1 = normalize_weighted_points(kp1, weights)
        norm_kp2, norm_transform2 = normalize_weighted_points(kp2, weights)

        # Construction of matrix A to find coefficients f_i of fundamental matrix F
        A = torch.cat((norm_kp1[:, :, 0].unsqueeze(-1) * norm_kp2,
                       norm_kp1[:, :, 1].unsqueeze(-1) * norm_kp2,
                       norm_kp2), -1)
        # Weighting each correspondence
        A = A * weights.unsqueeze(-1)

        F_estimate = []

        mask = torch.ones(3).to(kp1.device)
        mask[-1] = 0

        for batch in A:
            _, _, V = torch.svd(batch)
            # Solution to the least squares problem which is the singular vector
            # corresponding to the smallest singular value
            F = V[:, -1].view(3, 3)

            # Fundamental matrix is rank-deficient so we need to remove the least singular value
            U, S, V = torch.svd(F)
            F_proj = U @ torch.diag(S * mask) @ V.t()

            F_estimate.append(F_proj.unsqueeze(0))

        F_estimate = torch.cat(F_estimate, 0)
        F_estimate = norm_transform1.permute(0, 2, 1).bmm(F_estimate).bmm(norm_transform2)

        return F_estimate


class WeightEstimatorNet(nn.Module):

    def __init__(self, input_size):
        super(WeightEstimatorNet, self).__init__()

        self.estimator = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=1),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(),

            nn.Conv1d(128, 1024, kernel_size=1),
            nn.InstanceNorm1d(1024, affine=True),
            nn.LeakyReLU(),

            nn.Conv1d(1024, 512, kernel_size=1),
            nn.InstanceNorm1d(512, affine=True),
            nn.LeakyReLU(),

            nn.Conv1d(512, 256, kernel_size=1),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(),

            nn.Conv1d(256, 1, kernel_size=1),

            nn.Softmax(-1)
        )

    def forward(self, x):
        return self.estimator(x).squeeze(1)


class NormalizedEightPointNet(nn.Module):

    def __init__(self, num_iter):
        super(NormalizedEightPointNet, self).__init__()
        self.num_iter = num_iter

        self.estimator = ModelEstimator()

        self.weights_init = WeightEstimatorNet(4)
        self.weights_iter = WeightEstimatorNet(6)

    def forward(self, kp1, kp2, additional_info):
        """
        :param kp1: B x N x 2
        :param kp2: B x N x 2
        """
        # Normalize points to range [-1, 1]
        kp1, norm_transform1 = normalize_points(to_homogeneous_coordinates(kp1))
        kp2, norm_transform2 = normalize_points(to_homogeneous_coordinates(kp2))

        vectors_init = torch.cat(((kp1[:, :, :2] + 1) / 2, (kp2[:, :, :2] + 1) / 2), 2).permute(0, 2, 1)
        weights = self.weights_init(vectors_init)  # B x N

        F_estimate_init = self.estimator(kp1, kp2, weights)
        F_estimates = [F_estimate_init]

        for _ in range(1, self.num_iter):
            residuals = robust_symmetric_epipolar_distance(kp1, kp2, F_estimate_init).unsqueeze(1)

            vectors_iter = torch.cat((vectors_init, weights.unsqueeze(1), residuals), 1)
            weights = self.weights_iter(vectors_iter)

            F_estimate_iter = self.estimator(kp1, kp2, weights)
            F_estimates.append(F_estimate_iter)

        return F_estimates, norm_transform1, norm_transform2


class NetVGG(nn.Module):

    def __init__(self, grid_size, descriptor_size, nms_ks, verbose):
        super().__init__()

        self.grid_size = grid_size
        self.descriptor_size = descriptor_size
        self.nms_ks = nms_ks
        self.verbose = verbose

        self.conv1, self.score1 = make_vgg_ms_block(1, 64, 1)
        self.conv2, self.score2 = make_vgg_ms_block(64, 64, 1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3, self.score3 = make_vgg_ms_block(64, 64, 2)
        self.conv4, self.score4 = make_vgg_ms_block(64, 64, 2)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5, self.score5 = make_vgg_ms_block(64, 128, 4)
        self.conv6, self.score6 = make_vgg_ms_block(128, 128, 4)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7, self.score7 = make_vgg_ms_block(128, 128, 8)
        self.conv8, self.score8 = make_vgg_ms_block(128, 128, 8)

        self.detector = make_vgg_ms_detector(128, 256, self.grid_size)
        self.descriptor = make_vgg_ms_descriptor(128, 256, self.descriptor_size)

    def forward(self, x):
        """
        :param x: B x C x H x W
        """
        x = self.conv1(x)
        s1 = self.score1(x)

        x = self.conv2(x)
        s2 = self.score2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        s3 = self.score3(x)

        x = self.conv4(x)
        s4 = self.score4(x)

        x = self.pool2(x)

        x = self.conv5(x)
        s5 = self.score5(x)

        x = self.conv6(x)
        s6 = self.score6(x)

        x = self.pool3(x)

        x = self.conv7(x)
        s7 = self.score7(x)

        x = self.conv8(x)
        s8 = self.score8(x)

        s9 = self.detector(x)

        multi_scale_scores = torch.cat((s1, s2, s3, s4, s5, s6, s7, s8, s9), dim=1)
        nms_scores = multi_scale_nms(multi_scale_scores, self.nms_ks)
        score = multi_scale_softmax(nms_scores)

        desc = self.descriptor(x)
        desc = normalize(desc)

        return score, desc
