import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import APPNP


def model_instantiation(args):
    # model instantiation with bulit-in classes
    name = args.base_model.upper()

    if name == 'GCN':
        model = GCNbase
    elif name == 'APPNP':
        model = APPNPbase
    else:
        raise ValueError('Invalid model name')
    
    return model(args)


class GCNbase(torch.nn.Module):
    def __init__(self, args):
        super(GCNbase, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.num_classes)
        self.dp = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dp)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class APPNPbase(torch.nn.Module):
    def __init__(self, args):
        super(APPNPbase, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dp = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training, p=self.dp)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training, p=self.dp)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1)