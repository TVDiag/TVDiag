
from torch import nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import Sequential
from core.model.backbone.gatv2 import GATEncoder
import dgl.nn.pytorch as dglnn

class Encoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 graph_hidden_dim, 
                 out_dim,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 device='cpu'):
        super(Encoder, self).__init__()


        # feature aggregation
        self.graph_encoder = GATEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=2,
            heads=[8,1],
            feat_drop=feat_drop,
            attn_drop=attn_drop
        ).to(device)


    def forward(self, g, x):
        # h = self.sequential_encoder(x)
        f = self.graph_encoder(g, x)
        return f