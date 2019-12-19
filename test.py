from argparse import ArgumentParser
import cv2
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from source.nn.model import NormalizedEightPointNet
from source.datasets.colmap_dataset import ColmapBinDataset

from source.utils.eval_utils import evaluate
from source.utils.transform_utils import transform_F_to_image_space


def test(test_path, model_path, device):
    test_dataset = ColmapBinDataset(test_path, 20, None)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = NormalizedEightPointNet(num_iter=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    num_batches = 0

    net_accuracy = 0
    net_f1_score = 0
    net_time = 0

    ransac_accuracy = 0
    ransac_f1_score = 0
    ransac_time = 0

    for kp1, kp2, F, additional_info, _ in test_loader:
        kp1 = kp1.to(device)
        kp2 = kp2.to(device)
        F = F.to(device)
        additional_info = additional_info.to(device)

        num_batches += 1

        # # NN estimation
        forward_start_time = torch.cuda.Event(enable_timing=True)
        forward_end_time = torch.cuda.Event(enable_timing=True)

        forward_start_time.record()
        F_estimates, norm_transform1, norm_transform2 = model(kp1, kp2, additional_info)
        forward_end_time.record()
        torch.cuda.synchronize()

        el_time = forward_start_time.elapsed_time(forward_end_time)

        F_estimate = transform_F_to_image_space(norm_transform1, norm_transform2, F_estimates[-1]).detach()
        accuracy, f1_score = evaluate(kp1.cpu(), kp2.cpu(), F_estimate.cpu(), F.cpu())

        net_accuracy += accuracy
        net_f1_score += f1_score
        net_time += el_time

        # RANSAC estimation
        start_time = time.perf_counter()
        F_estimate = cv2.findFundamentalMat(kp1.cpu()[0].numpy(), kp2.cpu()[0].numpy(), method=cv2.FM_RANSAC, ransacReprojThreshold=1, confidence=0.9)
        end_time = time.perf_counter()

        el_time = (end_time - start_time) * 1000

        accuracy, f1_score = evaluate(kp1.cpu(), kp2.cpu(), F_estimate[0][np.newaxis].transpose((0, 2, 1)), F.cpu())

        ransac_accuracy += accuracy
        ransac_f1_score += f1_score
        ransac_time += el_time

    print("NN results")
    print(f"Accuracy: {net_accuracy / num_batches}")
    print(f"F1-score: {net_f1_score / num_batches}")
    print(f"Time: {net_time / num_batches} ms", end='\n\n')

    print("RANSAC results")
    print(f"Accuracy: {ransac_accuracy / num_batches}")
    print(f"F1-score: {ransac_f1_score / num_batches}")
    print(f"Time: {ransac_time / num_batches} ms")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()

    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test(args.test_path, args.model_path, _device)
