import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def dotsimilarity(x1, x2):
    x1_flat = torch.flatten(x1)
    x2_flat = torch.flatten(x2)

    return torch.dot(x1_flat, x2_flat)


def SimLoss (self, projs_u1, projs_u2, batch_size, temparature):
        # unlabeled_train_iter = iter(unlabel_loader)
        sim = nn.CosineSimilarity(dim=-1)
        u1s = F.normalize(projs_u1, dim=1)
        u2s = F.normalize(projs_u2, dim=1)
        projs_cat = torch.cat((u1s, u2s), dim=0)
        sim_mat = sim(u1s.unsqueeze(1), u2s.unsqueeze(0))
        

    


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.similarity = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(torch.device('cuda'))

    def forward(self, pred_u1, pred_u2):
        # add loss function here
        pred_u1 = F.normalize(pred_u1, dim=1)
        pred_u2 = F.normalize(pred_u2, dim=1)
        pred_concat = torch.cat((pred_u1, pred_u2), dim=0)
        similarity_matrix = self.similarity(pred_concat.unsqueeze(1), pred_concat.unsqueeze(0))

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(torch.device('cuda')).long()
        loss = self.criterion(logits, labels)

        return loss / (2*self.batch_size)