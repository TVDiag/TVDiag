import torch.nn as nn
from torch.functional import F
import dgl.nn.pytorch as dglnn

class GATEncoder(nn.Module):
    def __init__(self, 
                 in_dim,
                 hidden_dim, 
                 out_dim,
                 num_layers=3, 
                 heads=[8,8,1], 
                 feat_drop=0.5, 
                 attn_drop=0.5):
        super(GATEncoder, self).__init__()

        self.num_layers=num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.gatv2_layers.append(
            dglnn.GATv2Conv(
                in_feats=in_dim,
                out_feats=hidden_dim,
                num_heads=heads[0],
                residual=False,
                activation=self.activation,
                bias=True,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(0, num_layers-2):
            self.gatv2_layers.append(
                dglnn.GATv2Conv(
                    in_feats=hidden_dim * heads[l + 1],
                    out_feats=hidden_dim,
                    num_heads=heads[l],
                    activation=self.activation,
                    bias=True,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            dglnn.GATv2Conv(
                in_feats=hidden_dim * heads[-2],
                out_feats=out_dim,
                num_heads=heads[-1],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                activation=None,
                bias=True,
                share_weights=True,
            )
        )
        self.pool=dglnn.MaxPooling()

    def forward(self, g, x):
        h = x
        for l in range(self.num_layers-1):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gatv2_layers[-1](g, h).mean(1)
        return self.pool(g, logits)
