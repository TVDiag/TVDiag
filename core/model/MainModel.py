import torch
from torch import nn

from core.model.Classifier import Classifyer
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.args = args

        self.metric_encoder = Encoder(in_dim=args['metric_embedding_dim'],
                                      hiddens=args['graph_hidden'],                                
                                      out_dim=args['graph_out'])
        self.trace_encoder = Encoder(in_dim=args['trace_embedding_dim'],
                                     hiddens=args['graph_hidden'],
                                      out_dim=args['graph_out'])
        self.log_encoder = Encoder(in_dim=args['log_embedding_dim'],
                                   hiddens=args['graph_hidden'],
                                   out_dim=args['graph_out'])
        fuse_dim = self.metric_encoder.out_dim + self.trace_encoder.out_dim + self.log_encoder.out_dim

        self.locator = Classifyer(in_dim=fuse_dim, 
                                  hiddens=args['linear_hiddens'],
                                  out_dim=args['N_I'])
        self.typeClassifier = Classifyer(in_dim=fuse_dim,
                                         hiddens=args['linear_hiddens'],
                                         out_dim=args['N_A'])

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