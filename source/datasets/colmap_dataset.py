import numpy as np

from torch.utils.data import Dataset

import sqlite3
from source.utils.colmap_utils import read_cameras_binary, read_images_binary, \
    pair_id_to_image_ids, get_camera, \
    compose_fundamental_matrix, compute_residual


class ColmapBinDataset(Dataset):

    def __init__(self, dataset_root, num_min_matches, num_points, length=None):
        self.dataset = []

        self.num_points = num_points
        self.length = length

        cameras = read_cameras_binary(f"{dataset_root}/sparse/0/cameras.bin")
        images = read_images_binary(f"{dataset_root}/sparse/0/images.bin")

        connection = sqlite3.connect(f"{dataset_root}/reconstruction.db")
        cursor = connection.cursor()

        cursor.execute("SELECT pair_id, data FROM matches WHERE rows>=?;", (num_min_matches,))

        for row in cursor:
            img1_id, img2_id = pair_id_to_image_ids(row[0])

            img1 = images[img1_id]
            img2 = images[img2_id]

            # Get cameras
            K1, T1 = get_camera(img1, cameras)
            K2, T2 = get_camera(img2, cameras)

            F = compose_fundamental_matrix(K1, T1, K2, T2)

            matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)

            inner_cursor = connection.cursor()
            inner_cursor.execute("SELECT data, cols FROM keypoints WHERE image_id=?;", (img1_id,))

            inner_row = next(inner_cursor)
            kp1 = np.fromstring(inner_row[0], dtype=np.float32).reshape(-1, inner_row[1])

            inner_cursor.execute("SELECT data, cols FROM keypoints WHERE image_id=?;", (img2_id,))

            inner_row = next(inner_cursor)
            kp2 = np.fromstring(inner_row[0], dtype=np.float32).reshape(-1, inner_row[1])

            inner_cursor.execute("SELECT data FROM descriptors WHERE image_id=?;", (img1_id,))

            inner_row = next(inner_cursor)
            descriptor1 = np.float32(np.fromstring(inner_row[0], dtype=np.uint8).reshape(-1, 128))

            inner_cursor.execute("SELECT data FROM descriptors WHERE image_id=?;", (img2_id,))

            inner_row = next(inner_cursor)
            descriptor2 = np.float32(np.fromstring(inner_row[0], dtype=np.uint8).reshape(-1, 128))

            kp1 = kp1[matches[:, 0]]
            kp2 = kp2[matches[:, 1]]

            angle1 = kp1[:, 3]
            angle2 = kp2[:, 3]

            descriptor1 = descriptor1[matches[:, 0]]
            descriptor2 = descriptor2[matches[:, 1]]

            desc_dist = np.sqrt(np.mean((descriptor1 - descriptor2) ** 2, 1))[..., None]
            rel_scale = np.abs(kp1[:, 2] - kp2[:, 2])[..., None]
            rel_orient = np.minimum(np.abs(angle1 - angle2), np.abs(angle2 - angle1))[..., None]

            additional_info = np.hstack((desc_dist, rel_scale, rel_orient))

            kp1 = kp1[:, :2]
            kp2 = kp2[:, :2]

            res = compute_residual(kp1, kp2, F.T)
            residual_mask = res < 1

            if np.sum(residual_mask) >= num_min_matches:
                self.dataset.append([kp1, kp2, F.T, additional_info, residual_mask])

        cursor.close()
        connection.close()

    def __getitem__(self, idx):
        kp1, kp2, F, additional_info, residual_mask = self.dataset[idx]

        if self.num_points is not None:
            while kp1.shape[0] < self.num_points:
                perm = np.random.permutation(kp1.shape[0])[:self.num_points - kp1.shape[0]]

                kp1 = np.concatenate((kp1, kp1[perm]), 0)
                kp2 = np.concatenate((kp2, kp2[perm]), 0)

                additional_info = np.concatenate((additional_info, additional_info[perm]), 0)
                residual_mask = np.concatenate((residual_mask, residual_mask[perm]), 0)

        additional_info /= np.amax(additional_info, 0)

        if self.num_points is not None:
            if kp1.shape[0] > self.num_points:
                perm = np.random.permutation(kp1.shape[0])[: self.num_points]

                kp1 = kp1[perm]
                kp2 = kp2[perm]
                additional_info = additional_info[perm]
                residual_mask = residual_mask[perm]

        return kp1, kp2, F, additional_info, residual_mask

    def __len__(self):
        if self.length is None:
            return len(self.dataset)
        else:
            return self.length
