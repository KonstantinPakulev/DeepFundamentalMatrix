import torch

from source.utils.math_utils import calculate_distance_matrix, calculate_inv_similarity_matrix


def match_score(w_kp1, kp2, wv_kp1_mask, wv_kp2_mask, kp_dist_thresholds, kp1_desc, kp2_desc, verbose=False):
    """
    :param w_kp1: B x N x 2; keypoints on the first image projected to the second
    :param kp2: B x N x 2; keypoints on the second image
    :param wv_kp1_mask: B x N; keypoints on the first image which are visible on the second
    :param wv_kp2_mask: B x N; keypoints on the second image which are visible on the first
    :param kp_dist_thresholds: float or torch.tensor
    :param kp1_desc: B x N x C; descriptors for keypoints on the first image
    :param kp2_desc: B x N x C; descriptors for keypoints on the second image
    :param verbose: bool
    """
    b, n = wv_kp1_mask.shape

    # Calculate pairwise similarity measure
    desc_sim = calculate_inv_similarity_matrix(kp1_desc, kp2_desc)

    nn_desc_values, nn_desc_ids = desc_sim.min(dim=-1)

    # Get permutation of nn_desc_ids according to decrease of nn_desc_values
    nn_perm = nn_desc_values.argsort(dim=-1, descending=True)
    perm_nn_ids = torch.gather(torch.arange(n).view(1, -1).repeat(b, 1).to(nn_perm.device), -1, nn_perm)
    perm_nn_desc_ids = torch.gather(nn_desc_ids, -1, nn_perm)

    # Remove duplicate matches in each scene
    unique_match_mask = torch.zeros_like(wv_kp1_mask)

    for i, b_nn_desc_ids in enumerate(perm_nn_desc_ids):
        # Find unique elements in each scene
        b_unique, b_inv_indices = torch.unique(b_nn_desc_ids, sorted=False, return_inverse=True)

        # Restore forward mapping
        b_indices = torch.zeros_like(b_unique)
        b_indices[b_inv_indices] = perm_nn_ids[i]

        # Create unique match mask
        unique_match_mask[i][b_indices] = 1

    # Calculate pairwise keypoints distances
    kp_dist = calculate_distance_matrix(w_kp1, kp2)

    # Remove points that are not visible in both scenes by making their distance larger than maximum
    max_dist = kp_dist.max() * 2
    kp_dist = kp_dist + (1 - wv_kp1_mask.float().view(b, n, 1)) * max_dist
    kp_dist = kp_dist + (1 - wv_kp2_mask.float().view(b, 1, n)) * max_dist

    # Retrieve correspondent keypoints
    nn_kp_values = torch.gather(kp_dist, -1, nn_desc_ids.view(b, n, 1)).view(b, n)

    # Select minimum number of visible points for each scene
    # Repeatability-like approach
    # v1 = wv_kp1_mask.sum(dim=-1)
    # v2 = wv_kp2_mask.sum(dim=-1)
    # num_gt_corr = torch.cat([v1.view(-1, 1), v2.view(-1, 1)], dim=-1).min(dim=-1)[0].float()
    # LF-Net-like approach
    num_gt_corr = wv_kp2_mask.sum(dim=-1).float()

    match_score_list = torch.empty_like(kp_dist_thresholds)

    if verbose:
        num_matches_list = torch.empty_like(kp_dist_thresholds)
        num_gt_corrs = num_gt_corr.mean()
        kp_matches_list = torch.zeros([*kp_dist_thresholds.shape, *unique_match_mask.shape])

    for i, threshold in enumerate(kp_dist_thresholds):
        # Threshold correspondences
        desc_matches = nn_kp_values.le(threshold) * unique_match_mask

        # Calculate number of matches for each scene
        num_matches = desc_matches.sum(dim=-1).float()

        m_score = num_matches.float() / num_gt_corr.float()
        match_score_list[i] = m_score.mean()

        if verbose:
            num_matches_list[i] = num_matches.mean()
            kp_matches_list[i] = desc_matches

    if verbose:
        return match_score_list, num_matches_list, num_gt_corrs, nn_desc_ids, kp_matches_list
    else:
        return match_score_list
