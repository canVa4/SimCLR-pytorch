import torch
import numpy as np


class MarginTripletLoss(torch.nn.Module):
    def __init__(self, device, batch_size, semi_hard, temperature_or_m, use_cosine_similarity):
        super(MarginTripletLoss, self).__init__()
        self.batch_size = batch_size
        self.m_param = temperature_or_m
        self.device = device
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.semi_hard = semi_hard
        # self.ReLU = torch.nn.ReLU()

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

        mid = similarity_matrix[self.mask_samples_from_same_repr]
        negatives = mid.view(2 * self.batch_size, -1)
        zero = torch.zeros(1)
        triplet_matrix = torch.max(zero, negatives - positives + self.m_param)
        # 2N,2N-2 每一行代表了对于一个z关于其正类（z+batch）和其他反类的triplet loss
        # triplet_matrix = self.ReLU(negatives - positives + self.m_param)

        if self.semi_hard == True:
            # semi-hard
            semi_hard = - negatives + positives + self.m_param
            # print(semi_hard)
            semi_hard_mask = torch.max(semi_hard, zero).type(torch.bool)
            # print(semi_hard_mask)
            triplet_matrix_sh = triplet_matrix[semi_hard_mask]
            shape = triplet_matrix_sh.shape[0]
            # print(shape)
            # print(triplet_matrix)
            # print(triplet_matrix_sh)
            loss = torch.sum(triplet_matrix_sh)
            # print(loss/shape)
            return loss/shape
        else:
            loss = torch.sum(triplet_matrix)     # max( sim(neg) - sim(pos) + m, 0)
            return loss / (2*self.batch_size*(2*self.batch_size - 2))


if __name__ == "__main__":
    Loss = MarginTripletLoss('cuda', 3, True, 0.1, True)
    print(Loss.mask_samples_from_same_repr)
    xi = torch.randn((3, 3))
    xj = torch.randn((3, 3))

    loss = Loss(xi, xj)
    print(loss)
