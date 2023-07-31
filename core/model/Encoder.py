from core.model.backbone.GAT import *
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import Sequential


class Encoder(nn.Module):
    def __init__(self, in_dim, hiddens, out_dim, device='cpu'):
        super(Encoder, self).__init__()

        self.net = Sequential(
            dglnn.TAGConv(in_dim, hiddens, activation=torch.relu).to(device),
            dglnn.TAGConv(hiddens, out_dim, activation=torch.relu).to(device),
            dglnn.MaxPooling()
        )
        self.out_dim = out_dim

    def forward(self, g, x):
        f = self.net(g, x)
        return f
