import torch
from torch import nn

from core.model.Classifier import Classifyer
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.args = args

        self.metric_encoder = Encoder(in_dim=args.embedding_dim,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out)
        self.trace_encoder = Encoder(in_dim=args.embedding_dim,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out)
        self.log_encoder = Encoder(in_dim=args.embedding_dim,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out)
        fuse_dim = 3 * args.graph_out

        self.locator = Classifyer(in_dim=fuse_dim, 
                                  hiddens=args.linear_hidden,
                                  out_dim=args.N_I)
        self.typeClassifier = Classifyer(in_dim=fuse_dim,
                                         hiddens=args.linear_hidden,
                                         out_dim=args.N_T)

    def forward(self, batch_graphs):
        x_m = batch_graphs.ndata['metrics']
        x_t = batch_graphs.ndata['traces']
        x_l = batch_graphs.ndata['logs']
        
        f_m = self.metric_encoder(batch_graphs, x_m)
        f_t = self.trace_encoder(batch_graphs, x_t)
        f_l = self.log_encoder(batch_graphs, x_l)

        f = torch.cat((f_m, f_t, f_l), dim=1)
        root_logit = self.locator(f)
        type_logit = self.typeClassifier(f)

        return (f_m, f_t, f_l), root_logit, type_logit