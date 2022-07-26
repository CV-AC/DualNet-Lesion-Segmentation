import torch


class ReverseSampler:
    def __init__(self, feature_vector, label_vector, device, p=None):
        self.device = device
        self.feature_vector = feature_vector
        self.label_vector = label_vector

        if p:
            self.p = p
        else:
            self.weights = self.get_weights(self.label_vector)
            self.p = self.get_possibility()

        self.pos_index_list, self.neg_index_list = self.get_index_list()

    def get_weights(self, label):
        mask = (label > 0.5).float()
        b, c = mask.shape
        num = b * c
        num_pos = mask.sum()
        num_neg = num - num_pos

        weights = torch.tensor([(1. * num / num_neg).item(), (1. * num / num_pos).item()])
        weights = weights.to(self.device)
        return weights

    def get_possibility(self):
        p = self.weights[1] / torch.sum(self.weights)
        p = p.to(self.device)
        return p

    def get_index_list(self):
        pos_indexs = torch.nonzero(self.label_vector > 0.5)[:, 0]
        neg_indexs = torch.nonzero(self.label_vector <= 0.5)[:, 0]

        return pos_indexs.to(self.device), neg_indexs.to(self.device)

    def resample(self):
        b, c = self.feature_vector.shape

        pos_num = int(self.p * b)
        neg_num = b - pos_num

        pos_slice_index = torch.randint(0, self.pos_index_list.shape[0], (pos_num,)).to(self.device)
        neg_slice_index = torch.randint(0, self.neg_index_list.shape[0], (neg_num,)).to(self.device)

        slices = torch.cat((self.pos_index_list[pos_slice_index],
                            self.neg_index_list[neg_slice_index]), dim=0)

        re_feature = self.feature_vector[slices, :]
        re_label = self.label_vector[slices, :]

        return re_feature, re_label


def _to_vector(x):
    x = x.transpose(1, 3)
    x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
    return x
