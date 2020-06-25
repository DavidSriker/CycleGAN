import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    ''' Soft Dice Loss '''
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, gt, pred):
        smooth = 1.
        logits = torch.sigmoid(pred)
        iflat = logits.view(-1)
        tflat = gt.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class InvSoftDiceLoss(nn.Module):
    ''' Inverted Soft Dice Loss '''
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDiceLoss, self).__init__()

    def forward(self, gt, pred):
        smooth = 1.
        logits = torch.sigmoid(pred)
        iflat = 1 - logits.view(-1)
        tflat = 1 - gt.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))