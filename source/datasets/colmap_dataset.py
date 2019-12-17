import numpy as np

from torch.utils.data import Dataset

import sqlite3
from source.utils.colmap_utils import read_cameras_binary, read_images_binary, \
    pair_id_to_image_ids, get_camera, \
    compose_fundamental_matrix, compute_residual


class ColmapDataset(Dataset):

    def __init__(self, dataset_root, num_min_matches, num_points):
        self.kp1_list = []
        self.kp2_list = []

        self.F_list = []

        self.additional_info_list = []

        self.num_points = num_points

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
            K1, T1, sz1 = get_camera(img1, cameras)
            K2, T2, sz2 = get_camera(img2, cameras)

            F = compose_fundamental_matrix(K1, T1, K2, T2)

            matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)

            cursor2 = connection.cursor()
            cursor2.execute("SELECT data, cols FROM keypoints WHERE image_id=?;", (img1_id,))
            row2 = next(cursor2)
            keypoints1 = np.fromstring(row2[0], dtype=np.float32).reshape(-1, row2[1])

            cursor2.execute("SELECT data, cols FROM keypoints WHERE image_id=?;", (img2_id,))
            row2 = next(cursor2)
            keypoints2 = np.fromstring(row2[0], dtype=np.float32).reshape(-1, row2[1])

            cursor2.execute("SELECT data FROM descriptors WHERE image_id=?;", (img1_id,))
            row2 = next(cursor2)
            descriptor1 = np.float32(np.fromstring(row2[0], dtype=np.uint8).reshape(-1, 128))

            cursor2.execute("SELECT data FROM descriptors WHERE image_id=?;", (img2_id,))
            row2 = next(cursor2)
            descriptor2 = np.float32(np.fromstring(row2[0], dtype=np.uint8).reshape(-1, 128))

            angle1 = keypoints1[matches[:, 0], 3]
            angle2 = keypoints2[matches[:, 1], 3]

            desc_dist = np.sqrt(np.mean((descriptor1[matches[:, 0]] - descriptor2[matches[:, 1]]) ** 2, 1))[..., None]
            rel_scale = np.abs(keypoints1[matches[:, 0], 2] - keypoints2[matches[:, 1], 2])[..., None]
            rel_orient = np.minimum(np.abs(angle1 - angle2), np.abs(angle2 - angle1))[..., None]

            additional_info = np.hstack((desc_dist, rel_scale, rel_orient))

            keypoints1 = keypoints1[matches[:, 0], :2]
            keypoints2 = keypoints2[matches[:, 1], :2]

            res = compute_residual(keypoints1, keypoints2, F.T)

            if np.sum(np.uint8(res < 1)) >= num_min_matches:
                self.kp1_list.append(keypoints1)
                self.kp2_list.append(keypoints2)

                self.F_list.append(F.T)

                self.additional_info_list.append(additional_info)

        cursor.close()
        connection.close()

    def __getitem__(self, idx):
        kp1 = self.kp1_list[idx]
        kp2 = self.kp2_list[idx]

        F = self.F_list[idx]

        additional_info = self.additional_info_list[idx]

        while kp1.shape[0] < self.num_points:
            num_missing = self.num_points - kp1.shape[0]
            perm = np.random.permutation(kp1.shape[0])[:num_missing]

            kp1 = np.concatenate((kp1, kp1[perm]), 0)
            kp2 = np.concatenate((kp2, kp2[perm]), 0)

            additional_info_perm = additional_info[perm]
            additional_info = np.concatenate((additional_info, additional_info_perm), 0)

        additional_info /= np.amax(self.additional_info_list[idx], 0)

        if kp1.shape[0] > self.num_points:
            perm = np.random.permutation(kp1.shape[0])[: self.num_points]

            kp1 = kp1[perm]
            kp2 = kp2[perm]
            additional_info = additional_info[perm]

        return kp1, kp2, F, additional_info

    def __len__(self):
        return len(self.F_list)
