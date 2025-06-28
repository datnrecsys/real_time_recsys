import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight=4, neg_weight=1):
        super(WeightedMSELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):
        weight = torch.where(
            target > 0,
            torch.ones_like(pred) * self.pos_weight,
            torch.ones_like(pred) * self.neg_weight
        )
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * weight
        return loss.mean()