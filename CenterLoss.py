import torch
import torch.nn as nn
from torch.autograd import Variable, Function

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight = 0.01):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, y, feat):
        hist = Variable(torch.histc(y.cpu().data.float(), bins = self.num_classes, min = 0, max = self.num_classes)).cuda()
        feat = feat.view(feat.size()[0], -1)
        centers_pred = self.centers.index_select(0, y.long())
        diff = feat - centers_pred
        feat_mean = torch.Tensor().cuda()
        for i in range(self.num_classes):
            if i not in y.data:
                feat_mean = torch.cat((feat_mean, torch.zeros(1, self.feat_dim).cuda()), 0)
            else:
                feat_mean = torch.cat((feat_mean, (feat.index_select(0, Variable((y.data==i).nonzero().squeeze_(1)))).mean(0).data.unsqueeze_(0)), 0)
        centers_grad = Variable((hist / (1 + hist)).data.unsqueeze_(1)) * (self.centers - Variable(feat_mean))
        loss = self.loss_weight * 1 / 2.0 * diff.pow(2).sum(1).sum()
        return loss, centers_grad
