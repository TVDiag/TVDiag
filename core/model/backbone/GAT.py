import torch
from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling
from torch import nn
from torch.nn import BatchNorm1d
import dgl.nn.pytorch as dglnn


class GraphModel(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64, 128], device='cpu', attn_head=4, activation=0.2, **kwargs):
        super(GraphModel, self).__init__()
        '''
        Params:
            in_dim: the feature dim of each node
        '''
        layers = []
        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            dropout = kwargs["attn_drop"] if "attn_drop" in kwargs else 0
            layers.append(GATv2Conv(in_feats, out_feats=hidden, num_heads=attn_head,
                                    attn_drop=dropout, negative_slope=activation, allow_zero_in_degree=True))
            self.maxpool = nn.MaxPool1d(attn_head)

        # self.TAG = TAGClassifier(in_dim, 128).to(device)

        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))

    def forward(self, graph):
        """
        Input:
            x -- tensor float [batch_size*node_num, feature_in_dim] N = {s1, s2, s3, e1, e2, e3}
        """
        out = graph.ndata['metrics'].float()
        for layer in self.net:
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out)  # [bz*node, out_dim] --> [bz, out_dim]
        # return self.TAG(graph, out)
