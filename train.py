from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from source.nn.model import NormalizedEightPointNet
from source.datasets.colmap_dataset import ColmapBinDataset

from source.utils.math_utils import symmetric_epipolar_distance
from source.utils.transform_utils import to_homogeneous_coordinates
from source.utils.eval_utils import evaluate
from source.utils.transform_utils import transform_F_to_image_space


def train(train_path, val_path, device):
    train_dataset = ColmapBinDataset(train_path, 20, 1000)

    val_size = 100
    val_dataset = ColmapBinDataset(val_path, 20, None, val_size)

    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = NormalizedEightPointNet(num_iter=3)
    model.to(device)

    optimizer = optim.Adamax(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(98):
        train_epoch_loss = 0

        train_epoch_accuracy = 0
        train_epoch_f1_score = 0

        num_batches = 0

        for i, (kp1, kp2, F, additional_info, residuals_mask) in enumerate(train_loader):
            model.train()

            with torch.autograd.set_detect_anomaly(True):
                kp1 = kp1.to(device)
                kp2 = kp2.to(device)
                F = F.to(device)
                additional_info = additional_info.to(device)
                residuals_mask = residuals_mask.to(device)

                optimizer.zero_grad()

                F_estimates, norm_transform1, norm_transform2 = model(kp1, kp2, additional_info)

                kp1_norm = torch.bmm(to_homogeneous_coordinates(kp1), norm_transform1.permute(0, 2, 1))
                kp2_norm = torch.bmm(to_homogeneous_coordinates(kp2), norm_transform2.permute(0, 2, 1))

                loss = torch.tensor(0.0).to(device)

                for F_estimate in F_estimates:
                    # loss += symmetric_epipolar_distance(kp1_norm, kp2_norm, F_estimate).mean()
                    loss += (symmetric_epipolar_distance(kp1_norm, kp2_norm, F_estimate) * residuals_mask.float()).sum() / residuals_mask.sum().float()

                loss.backward()

                optimizer.step()

            model.eval()

            train_epoch_loss += loss.item()
            num_batches += 1

            F_estimate = transform_F_to_image_space(norm_transform1, norm_transform2, F_estimates[-1]).detach()
            accuracy, f1_score = evaluate(kp1.cpu(), kp2.cpu(), F_estimate.cpu(), F.cpu())

            train_epoch_accuracy += accuracy
            train_epoch_f1_score += f1_score

        print(f"Epoch {epoch}")
        print(f"Train average loss: {train_epoch_loss / num_batches}")
        print(f"Train average accuracy: {train_epoch_accuracy / num_batches}")
        print(f"Train average f1_score: {train_epoch_f1_score / num_batches}")

        val_epoch_accuracy = 0
        val_epoch_f1_score = 0

        for kp1, kp2, F, additional_info, _ in val_loader:
            kp1 = kp1.to(device)
            kp2 = kp2.to(device)
            F = F.to(device)
            additional_info = additional_info.to(device)

            F_estimates, norm_transform1, norm_transform2 = model(kp1, kp2, additional_info)

            F_estimate = transform_F_to_image_space(norm_transform1, norm_transform2, F_estimates[-1]).detach()
            accuracy, f1_score = evaluate(kp1.cpu(), kp2.cpu(), F_estimate.cpu(), F.cpu())

            val_epoch_accuracy += accuracy
            val_epoch_f1_score += f1_score

        print("Validation:")
        print(f"Val average accuracy: {val_epoch_accuracy / val_size}")
        print(f"Val average f1_score: {val_epoch_f1_score / val_size}")

        torch.save(model.state_dict(), f"checkpoints_sh/model_epoch{epoch % 3}.pt")
        scheduler.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)

    args = parser.parse_args()

    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(args.train_path, args.val_path, _device)
