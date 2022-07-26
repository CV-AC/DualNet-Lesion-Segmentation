import torch
import torch.nn as nn
from torch.nn import functional as F


class CriterionCBCE(nn.Module):
    def __init__(self, weight=1.):
        super(CriterionCBCE, self).__init__()
        self.weight = weight

    def forward(self, preds, edges):
        mask = (edges > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
        num_neg = c * h * w - num_pos
        weight = torch.zeros_like(mask)
        weight[edges > 0.5] = num_neg / (num_pos + num_neg)
        weight[edges <= 0.5] = num_pos / (num_pos + num_neg)

        loss = torch.nn.functional.binary_cross_entropy(preds.float(), edges.float(), weight=weight, reduction='none')
        loss = torch.sum(loss) / b
        loss = self.weight * loss
        return loss


class CriterionDice(nn.Module):
    def __init__(self):
        super(CriterionDice, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 1
        y_true_f = y_true.flatten().float()
        y_pred_f = y_pred.flatten().float()
        intersection = torch.sum(y_true_f * y_pred_f)
        dice_coef = 2 * (intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return 1 - dice_coef


# REF: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py
def focal_loss_with_logits(
        output,
        target,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        normalized=False,
        reduced_threshold=None,
        eps=1e-6):
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())

    logpt = F.binary_cross_entropy(output, target, reduction="none")
    pt = torch.exp(-logpt)

    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class CriterionFocal(nn.Module):
    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2.0,
                 ):
        super(CriterionFocal, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        loss = focal_loss_with_logits(y_pred, y_true, gamma=self.gamma, alpha=self.alpha)
        loss = self.weight * loss
        return loss


class CriterionScaledCBCE(nn.Module):
    def __init__(self, stages=2):
        super(CriterionScaledCBCE, self).__init__()
        self.stages = stages
        self.criterion = CriterionCBCE(weight=1.0)

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)
        if len(preds) >= 2 and self.stages == 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2 * 0.4
        elif len(preds) == 1 or self.stages == 1:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss = self.criterion(scale_pred, target)
            return loss


class CriterionScaledDice(nn.Module):
    def __init__(self, stages=2):
        super(CriterionScaledDice, self).__init__()
        self.stages = stages
        self.criterion = CriterionDice()

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)
        if len(preds) >= 2 and self.stages == 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2 * 0.4
        elif len(preds) == 1 or self.stages == 1:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss = self.criterion(scale_pred, target)
            return loss


class CriterionScaledFocal(nn.Module):
    def __init__(self,
                 stages=2,
                 alpha=0.25,
                 gamma=2.0):
        super(CriterionScaledFocal, self).__init__()
        self.stages = stages
        self.criterion = CriterionFocal(alpha=alpha, gamma=gamma)

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)
        if len(preds) >= 2 and self.stages == 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2 * 0.4
        elif len(preds) == 1 or self.stages == 1:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            scale_pred = torch.sigmoid(scale_pred)
            loss = self.criterion(scale_pred, target)
            return loss
