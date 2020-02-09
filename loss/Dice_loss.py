import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.smooth = 1

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "error: the batch-size of predict and target don't match"
        predict = predict.contiguous.view(predict.shape[0], -1)  # contiguous make tensor into a continuous distribution
        target = target.contiguous.view(target.shape[0], -1)
        dice_molecule = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        dice_denominator =torch.sum(predict.pow(2) + target.pow(2), dim=1) + self.smooth
        loss = 1 - dice_molecule / dice_denominator
        return loss

class Diceloss(nn.Module):
    def __init__(self, weights=None, **kwargs):
        super(Diceloss, self).__init__()
        self.weights = weights
        self.kwargs = kwargs

    def forward(self, predict, target):
        assert predict.shape == target.shape, "error: the shape of predict and target don't match"
        dice = loss(**self.kwargs)
        total_loss = 0.
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            dice_loss = dice(predict[:, i], target[:, i])
            if self.weights is not None:
                dice_loss *= self.weights[i]
            total_loss += dice_loss

        return total_loss / target.shape[1]