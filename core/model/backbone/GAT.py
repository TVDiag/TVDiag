import torch
from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling
from torch import nn
from torch.nn import BatchNorm1d
import dgl.nn.pytorch as dglnn


# class TAGClassifier(nn.Module):
#     """
#     两层TAGConv+最大池化+线性分类器
#     """
#
#     def __init__(self, in_dim, hidden_dim):
#         super(TAGClassifier, self).__init__()
#         self.conv1 = dglnn.TAGConv(in_dim, hidden_dim, activation=torch.relu)
#         self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim, activation=torch.relu)
#         # self.pool = dglnn.AvgPooling()
#         self.pool = dglnn.MaxPooling()
#         # self.classify = nn.Linear(hidden_dim, n_classes)
#
#     def forward(self, g, h):
#         h = self.conv1(g, h)
#         h = self.conv2(g, h)
#         h = self.pool(g, h)
#         return h
#
#     def get_embeds(self, g, h, pool=False):
#         h = self.conv1(g, h)
#         # h = F.dropout(h, p=0.5, training=True)
#         h = self.conv2(g, h)
#         # h = F.dropout(h, p=0.5, training=True)
#         if pool:
#             h = self.pool(g, h)
#         return h

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
