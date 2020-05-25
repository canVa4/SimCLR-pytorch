import torch
import numpy as np


class NTLogisticLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature_or_m, use_cosine_similarity):
        super(NTLogisticLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature_or_m
        self.device = device
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.sigmoid = torch.nn.Sigmoid()
        self.method = 2  # 0为无操作  1为under-sampling  2为change weight

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1) * -1

        logits_pos = self.sigmoid(positives / self.temperature)
        logits_neg = self.sigmoid(negatives / self.temperature)
        logits_pos = torch.log(logits_pos)
        logits_neg = torch.log(logits_neg)
        if self.method == 1:
            # under-sampling
            all_one_vec = np.ones((1, 2 * self.batch_size,))
            all_zero_vec = np.zeros((1, 2 * self.batch_size * (2 * self.batch_size - 3)))
            under_sampling_vec = np.column_stack((all_one_vec, all_zero_vec)).flatten()
            np.random.shuffle(under_sampling_vec)
            under_sampling_matrix = torch.tensor(under_sampling_vec).view(
                (2 * self.batch_size, 2 * self.batch_size - 2)).type(torch.bool).to(self.device)

            logits_neg = logits_neg[under_sampling_matrix]
            loss = torch.sum(logits_pos) + torch.sum(logits_neg)
            return -loss
        elif self.method == 2:
            neg_count = 2 * self.batch_size * (2 * self.batch_size - 2)
            pos_count = 2 * self.batch_size
            loss = neg_count * torch.sum(logits_pos) + pos_count * torch.sum(logits_neg)
            return -loss / (pos_count + neg_count)
        else:
            total_logits = torch.cat((logits_pos, logits_neg), dim=1)
            loss = torch.sum(total_logits)
            return -loss


if __name__ == "__main__":
    Loss = NTLogisticLoss('cuda', 3, 0.5, True)
    xi = torch.randn((3, 3))
    xj = torch.randn((3, 3))
    loss = Loss(xi, xj)

