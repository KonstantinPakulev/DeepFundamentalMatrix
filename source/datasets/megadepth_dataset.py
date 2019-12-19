import os
import pickle
import numpy as np

from torch.utils.data import Dataset

MIN_OVERLAP_RATIO = 0.5
MAX_OVERLAP_RATIO = 1.0
MAX_SCALE_RATIO = np.inf

from source.utils.colmap_utils import compose_fundamental_matrix, compute_residual


class MegaDepthGroundTruthDataset(Dataset):

    def __init__(self, scene_info_path, min_num_of_matches, length):
        self.dataset = []
        self.length = length

        dataset_path = os.path.join(scene_info_path, "gt_matches_dfe.p")

        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as file:
                self.dataset = pickle.load(file)
        else:
            scene_info_names = os.listdir(scene_info_path)

            # Use only few scenes, since there are too many of them
            for scene_info_name in scene_info_names[:3]:
                scene_info_path = os.path.join(scene_info_path, scene_info_name)

                if not os.path.exists(scene_info_path):
                    print("Scene path doesn't exist:", scene_info_path)
                    continue

                scene_info = np.load(scene_info_path, allow_pickle=True)

                overlap_matrix = scene_info['overlap_matrix']
                scale_ratio_matrix = scene_info['scale_ratio_matrix']

                valid = np.logical_and(
                    np.logical_and(overlap_matrix >= MIN_OVERLAP_RATIO,
                                   overlap_matrix <= MAX_OVERLAP_RATIO),
                    scale_ratio_matrix <= MAX_SCALE_RATIO)

                pairs = np.vstack(np.where(valid))

                points3D_id_to_2D = scene_info['points3D_id_to_2D']
                points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']

                # World-to-camera poses
                poses = scene_info['poses']

                # Camera intrinsics
                intrinsics = scene_info['intrinsics']

                for pair_idx in range(pairs.shape[1]):
                    idx1 = pairs[0, pair_idx]
                    idx2 = pairs[1, pair_idx]

                    matches = np.array(list(
                        points3D_id_to_2D[idx1].keys() &
                        points3D_id_to_2D[idx2].keys()
                    ))

                    # Scale filtering
                    matches_nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
                    matches_nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
                    scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
                    matches = matches[np.where(scale_ratio <= MAX_SCALE_RATIO)[0]]

                    # Image-to-image matches
                    kp1 = np.array([points3D_id_to_2D[idx1][m] for m in matches])
                    kp2 = np.array([points3D_id_to_2D[idx2][m] for m in matches])

                    camera1_pose = poses[idx1]
                    camera2_pose = poses[idx2]

                    camera1_intrinsics = intrinsics[idx1]
                    camera2_intrinsics = intrinsics[idx2]

                    F = compose_fundamental_matrix(camera1_intrinsics, camera1_pose, camera2_intrinsics, camera2_pose).T

                    residuals = compute_residual(kp1, kp2, F)
                    residuals_mask = residuals < 1

                    if np.sum(residuals_mask) >= min_num_of_matches:
                        self.dataset.append([kp1, kp2, F])

                with open(dataset_path, 'wb') as file:
                    pickle.dump(self.dataset, file)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        kp1, kp2, F = self.dataset[index]

        # while kp1.shape[0] < self.samples_per_batch:
        #     num_missing = self.samples_per_batch - kp1.shape[0]
        #     perm = np.random.permutation(kp1.shape[0])[:num_missing]
        #
        #     kp1 = np.concatenate((kp1, kp1[perm]), 0)
        #     kp2 = np.concatenate((kp2, kp2[perm]), 0)
        #
        #     residuals_mask = np.concatenate((residuals_mask, residuals_mask[perm]), 0)
        #
        # if kp1.shape[0] > self.samples_per_batch:
        #     perm = np.random.permutation(kp1.shape[0])[: self.samples_per_batch]
        #
        #     kp1 = kp1[perm]
        #     kp2 = kp2[perm]
        #
        #     residuals_mask = residuals_mask[perm]

        return kp1.astype(np.float32), kp2.astype(np.float32), F.astype(np.float32)
