import torch
import torch.nn.functional as F

from dual.combiner import Combiner
from dual.sampler import ReverseSampler
from loss.criterion import CriterionDice


class ToolboxCombiner(Combiner):
    def __init__(self, model, device, max_epoch_number, stop_alpha=0., p_rate=None):
        super(ToolboxCombiner, self).__init__(model, device, max_epoch_number, stop_alpha, p_rate=p_rate)
        self.criterion_dice = CriterionDice()
        self.global_iteration = 0

    def _to_vector(self, x):
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        return x

    def _restore(self, x, shape):
        x = x.reshape(shape[0], shape[2], shape[3], shape[1])
        x = x.transpose(2, 3)
        x = x.transpose(1, 2)
        return x

    def upsample(self, x, h, w):
        return F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)

    def reset_global_iteration(self, iteration):
        self.global_iteration = iteration

    def dual_forward(self, x, label, **kwargs):
        alpha = self.get_alpha()
        if "feature_cb" in kwargs:
            x = self.model(x, feature_cb=True)
        elif "feature_rb" in kwargs:
            alpha = 1 - alpha
            x = self.model(x, feature_rb=True)

        label = self._to_vector(label)
        x_dsn = self._to_vector(x[1])
        x = self._to_vector(x[0])
        loss2 = self.criterion_dice(x_dsn, label)

        if "feature_rb" in kwargs:
            x, label = ReverseSampler(x, label, self.device, p=self.p_rate).resample()
        loss1 = self.criterion_dice(x, label)
        loss = alpha * loss1 + loss2 * 0.2

        return loss

    def classfier(self, x):
        x = self.model(x, classify=True)
        x = torch.sigmoid(x)
        return x
