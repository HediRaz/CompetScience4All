import torch


class ROC_STAR_LOSS():

    def __init__(self, max_pos, max_neg, gamma):
        self.gamma = gamma
        self.iteration = 0
        self.p = torch.square
        self.max_pos = max_pos
        self.max_neg = max_neg

    def __call__(self, preds, labels):
        preds = preds[:, 1]
        nb_pos = torch.sum(labels)
        nb_neg = labels.shape[0] - nb_pos
        if nb_pos == 0 or nb_neg == 0:
            return 1e-8

        loss = 0

        pos_mask = labels==1
        max_pos = min(nb_pos, self.max_pos)
        max_neg = min(nb_neg, self.max_neg)

        d = preds[~pos_mask][:max_neg].repeat(nb_pos).view(nb_pos, max_neg) - preds[pos_mask].view(-1, 1).expand(-1, max_neg) + self.gamma
        d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
        loss += torch.sum(self.p(d[d>0]))/(max_neg*nb_pos)
    
        d = preds[~pos_mask].repeat(max_pos).view(max_pos, nb_neg) - preds[pos_mask][:max_pos].view(-1, 1).expand(-1, nb_neg) + self.gamma
        d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
        loss += torch.sum(self.p(d[d>0]))/(max_pos*nb_neg)

        return loss


class ROC_LOSS():

    def __init__(self, max_pos, max_neg):
        self.max_pos = max_pos
        self.max_neg = max_neg

    def __call__(self, preds, labels):
        preds = preds[:, 1]
        nb_pos = torch.sum(labels)
        nb_neg = labels.shape[0] - nb_pos
        if nb_pos == 0 or nb_neg == 0:
            return 1e-8

        loss = 0

        pos_mask = labels==1
        max_pos = min(nb_pos, self.max_pos)
        max_neg = min(nb_neg, self.max_neg)

        d =  preds[pos_mask].view(-1, 1).expand(-1, max_neg) - preds[~pos_mask][:max_neg].repeat(nb_pos).view(nb_pos, max_neg)
        loss += -torch.sum(torch.sigmoid(d))/(max_neg*nb_pos)
    
        d = preds[pos_mask][:max_pos].view(-1, 1).expand(-1, nb_neg) - preds[~pos_mask].repeat(max_pos).view(max_pos, nb_neg)
        loss += -torch.sum(torch.sigmoid(d))/(max_pos*nb_neg)

        return loss
