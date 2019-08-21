import torch
import torch.nn as nn


def wings_metric(output, target):
    score = nn.MSELoss(output, target)
    return score
