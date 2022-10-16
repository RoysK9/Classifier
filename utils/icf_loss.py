
import torch
import torch.nn.functional as F
from torch import nn,optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ICFLoss(nn.Module):
    def __init__(self, weight=None, num_list=None):
        super(ICFLoss, self).__init__()
        self.num_list = torch.FloatTensor(num_list)
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, input, target):
        class_num = []
        for ind in target:
            class_num.append(self.num_list[ind.item()])
        class_num = torch.FloatTensor(class_num).to(device)
        
        Loss = F.cross_entropy(input, target, reduction='none', weight=self.weight) / class_num
        return Loss.mean()


def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()


class IBLoss(nn.Module):
    def __init__(self, num_class, weight=None, alpha=10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.num_class = num_class

    def forward(self, input, target, features): # fearturesは全結合層の最終層への入力
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_class)),1) # N * 1
        
        features = torch.sum(torch.abs(features),1)
        ib = grads*features.reshape(-1)

        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)