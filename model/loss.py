import torch.nn as nn


def mse_loss(output, target):
    loss = nn.MSELoss(output, target)
    return loss


def find_best_orientation(kpts_gt, kpts_est, difference_thresh=0.5):
    """
    Same as flip_loss, but also returns the best orientation and the original vs min loss
    :param kpts_gt: Ground-truth keypoints
    :param kpts_est: Inferred keypoints
    :param difference_thresh: Minimal loss must be difference_thresh*original loss to be returned
    :return: Unflipped loss, flipped loss, recommended flipping operation
    """
    for batch in range(0, len(kpts_gt), 1):
        tmp = kpts_est[batch].clone()

        # no flip
        min_loss = mse_loss(kpts_gt[batch], tmp)
        original_loss = min_loss
        min_op = "identity"
        # horizontal flip
        tmp[0::2] = 1.0 - tmp[0::2]
        loss = mse_loss(kpts_gt[batch], tmp)
        if loss < (min_loss * difference_thresh):
            min_loss = loss
            min_op = "hor"
        # horizontal and vertical flip
        tmp[1::2] = 1.0 - tmp[1::2]
        loss = mse_loss(kpts_gt[batch], tmp)
        if loss < (min_loss * difference_thresh):
            min_loss = loss
            min_op = "hor_ver"
        # vertical flip
        tmp[0::2] = 1.0 - tmp[0::2]
        loss = mse_loss(kpts_gt[batch], tmp)
        if loss < (min_loss * difference_thresh):
            min_loss = loss
            min_op = "ver"

    return original_loss, min_loss, min_op


def flip_loss(kpts_gt, kpts_est, difference_thresh=0.5):
    """
    This loss is invariant to flipped images
    :param kpts_gt: Ground-truth keypoints
    :param kpts_est: Inferred keypoints
    :param difference_thresh: Minimal loss must be difference_thresh*original loss to be returned
    :return: MSELoss of the best flip
    """
    original_loss, min_loss, min_op = find_best_orientation(kpts_gt, kpts_est, difference_thresh)
    return min_loss