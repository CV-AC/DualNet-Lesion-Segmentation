import math
import torch

from dual.sampler import ReverseSampler


class Combiner:
    def __init__(self, model, device, max_epochs, stop_alpha=0., p_rate=None):
        self.device = device
        self.model = model
        self.max_epochs = max_epochs
        self.stop_alpha = stop_alpha

        self.p_rate = p_rate
        self.global_iteration = 0
        self.init_all_parameters()

    def init_all_parameters(self):
        self.div_epoch = self.max_epochs

        if self.stop_alpha > 0:
            self.div_epoch -= 1

    def reset_epoch(self, epoch):
        self.epoch = epoch + 1

    def reset_global_iteration(self, it):
        self.global_iteration = it

    def get_alpha(self):
        rate = math.sqrt(1 - self.stop_alpha)
        l = 1 - ((rate * self.epoch - 1) / self.div_epoch) ** 2
        if l < self.stop_alpha:
            l = self.stop_alpha

        return l

    def forward_share_features(self, output_a, label_a, output_b, label_b, stages=3):
        loss_a = sum([get_loss(preds, label_a) for preds in output_a])
        loss_b = sum([get_loss(preds, label_b) for preds in output_b])
        loss = (loss_a + loss_b) / (2. * stages)
        return loss

    def dual_mix_forward(self, image, label, sample_image, sample_label, stages=1):
        image_a, image_b = image.to(self.device), sample_image.to(self.device)
        label_a, label_b = label.to(self.device), sample_label.to(self.device)

        output_list_a, output_list_b = (
            self.model(image_a, feature_cb=True),
            self.model(image_b, feature_rb=True),
        )

        share_stages = 6 - stages
        share_loss = self.forward_share_features(output_list_a[:share_stages], label_a,
                                                 output_list_b[:share_stages], label_b, share_stages)

        alpha = self.get_alpha()
        output_list_a = output_list_a[-stages:]
        output_list_b = output_list_b[-stages:]
        label_a, label_b = self._to_vector(label_a), self._to_vector(label_b)

        losses = []
        for i in range(len(output_list_a)):
            output_a, output_b = self._to_vector(output_list_a[i]), self._to_vector(output_list_b[i])
            output_b, new_label_b = ReverseSampler(output_b, label_b, self.device, p=self.p_rate).resample()
            loss_cb = get_loss(output_a, label_a)
            loss_rb = get_loss(output_b, new_label_b)
            loss = alpha * loss_cb + (1 - alpha) * loss_rb
            losses.append(loss)

        loss = sum(losses) / stages
        loss = (loss + share_loss) / 2.
        return loss

    def _to_vector(self, x):
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        return x


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return 2 * (intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def get_loss(preds, edges, batch_size=1):
    losses = 1 - dice_coef(edges.float(), preds.float())
    loss = torch.sum(losses) / batch_size
    return loss
