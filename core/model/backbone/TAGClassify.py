import dgl.nn.pytorch as dglnn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class TAGClassifier(nn.Module):
    """
    两层TAGConv+最大池化+线性分类器
    """

    def __init__(self, in_dim, hidden_dim, n_classes):
        super(TAGClassifier, self).__init__()
        self.conv1 = dglnn.TAGConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim, activation=F.relu)
        # self.pool = dglnn.AvgPooling()
        self.pool = dglnn.MaxPooling()
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        x = self.conv1(g, h)
        x = self.conv2(g, x)
        x = self.pool(g, x)
        return self.classify(x)

    def get_embeds(self, g, h, pool=False):
        x = self.conv1(g, h)
        # x = F.dropout(x, p=0.5, training=True)
        x = self.conv2(g, x)
        # x = F.dropout(x, p=0.5, training=True)
        if pool:
            x = self.pool(g, x)
        return x