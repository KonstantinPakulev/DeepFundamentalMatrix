from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from source.datasets.colmap_dataset import ColmapDataset
from source.utils.math_utils import symmetric_epipolar_distance
from source.nn.model import NormalizedEightPointNet
from source.utils.transform_utils import to_homogeneous_coordinates
from source.utils.eval_utils import transform_F_into_image_space, compute_error


def train(dataset_path, device):
    dataset = ColmapDataset(dataset_path, 20, 1000)

    loader = DataLoader(dataset, batch_size=16)

    model = NormalizedEightPointNet(num_iter=3)
    model.to(device)

    optimizer = optim.Adamax(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(98):
        epoch_loss = 0

        epoch_accuracy = 0
        epoch_f1_score = 0

        num_batches = 0

        model.train()

        for i, (kp1, kp2, F, additional_info) in enumerate(loader):
            with torch.autograd.set_detect_anomaly(True):
                kp1 = kp1.to(device)
                kp2 = kp2.to(device)
                F = F.to(device)
                additional_info = additional_info.to(device)

                # Train part
                optimizer.zero_grad()

                F_estimates, norm_transform1, norm_transform2 = model(kp1, kp2, additional_info)

                kp1_norm = torch.bmm(to_homogeneous_coordinates(kp1), norm_transform1.permute(0, 2, 1))
                kp2_norm = torch.bmm(to_homogeneous_coordinates(kp2), norm_transform2.permute(0, 2, 1))

                loss = torch.tensor(0.0).to(device)

                for F_estimate in F_estimates:
                    loss += symmetric_epipolar_distance(kp1_norm, kp2_norm, F_estimate).mean()

                loss.backward()

                optimizer.step()

            model.eval()

            # Eval part
            epoch_loss += loss.item()
            num_batches += 1

            F_estimate_image = transform_F_into_image_space(norm_transform1, norm_transform2, F_estimates[-1]).detach()

            accuracy = 0
            f1_score = 0

            for j in range(kp1.shape[0]):
                j_a, j_f1 = compute_error(kp1[j].cpu(), kp2[j].cpu(), F_estimate_image[j].cpu(), F[j].cpu())

                accuracy += j_a
                f1_score += j_f1

            accuracy /= kp1.shape[0]
            f1_score /= kp1.shape[0]

            epoch_accuracy += accuracy
            epoch_f1_score += f1_score

        print(f"Epoch {epoch}")
        print(f"Average loss: {epoch_loss / num_batches}")
        print(f"Average accuracy: {epoch_accuracy / num_batches}")
        print(f"Average f1_score: {epoch_f1_score / num_batches}")

        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch % 3}.pt")

        scheduler.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)

    args = parser.parse_args()

    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(args.dataset_path, _device)
