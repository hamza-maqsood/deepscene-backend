# Contruct a two-layer GNN model
# import dgl.nn as dglnn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

#
# class SAGE(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats):
#         super().__init__()
#         self.conv1 = dglnn.SAGEConv(
#             in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
#         self.conv2 = dglnn.SAGEConv(
#             in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
#
#     def forward(self, graph, inputs):
#         # inputs are features of nodes
#         h = self.conv1(graph, inputs)
#         h = F.relu(h)
#         h = self.conv2(graph, h)
#         return h
#
#
# class MLPPredictor(nn.Module):
#     def __init__(self, in_features, out_classes):
#         super().__init__()
#         self.W = nn.Linear(in_features * 2, out_classes)
#
#     def apply_edges(self, edges):
#         h_u = edges.src['h']
#         h_v = edges.dst['h']
#         score = self.W(torch.cat([h_u, h_v], 1))
#         return {'score': score}
#
#     def forward(self, graph, h):
#         # h contains the node representations computed from the GNN defined
#         # in the node classification section (Section 5.1).
#         with graph.local_scope():
#             graph.ndata['h'] = h
#             graph.apply_edges(self.apply_edges)
#             return graph.edata['score']
#
#
# class DirectionModel(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super().__init__()
#         self.sage = SAGE(in_features, hidden_features, out_features)
#         self.pred = MLPPredictor(out_features, 6)
#
#     def forward(self, g, x):
#         h = self.sage(g, x)
#         return self.pred(g, h)
#
#
# class DistanceModel(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super().__init__()
#         self.sage = SAGE(in_features, hidden_features, out_features)
#         self.pred = MLPPredictor(out_features, 3)
#
#     def forward(self, g, x):
#         h = self.sage(g, x)
#         return self.pred(g, h)
#
#

import dgl.nn as dglnn
from dgl.nn import EGATConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 3, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v, self.temp_edge_feats], 1))
        return {'score': score}

    def forward(self, graph, h, e):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            self.temp_edge_feats = e
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class PlacementPredictor(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # self.sage = SAGE(in_features, hidden_features, out_features)
        self.egatcon = EGATConv(in_node_feats=in_features,
                                in_edge_feats=in_features,
                                out_node_feats=hidden_features,
                                out_edge_feats=hidden_features,
                                num_heads=1,)
        self.egatcon1 = EGATConv(in_node_feats=hidden_features,
                                in_edge_feats=hidden_features,
                                out_node_feats=30,
                                out_edge_feats=30,
                                num_heads=1,)
        self.egatcon2 = EGATConv(in_node_feats=30,
                                in_edge_feats=30,
                                out_node_feats=out_features,
                                out_edge_feats=out_features,
                                num_heads=1,)
        
        # self.pred = MLPPredictor(out_features, 6)

    def forward(self, g, x, e):
        hidden_node_feats1, hidden_edge_feats1 = self.egatcon(g, x, e)
        hidden_node_feats2, hidden_edge_feats2 = self.egatcon1(g, hidden_node_feats1, hidden_edge_feats1)
        out_node_feats, out_edge_feats = self.egatcon2(g, hidden_node_feats2, hidden_edge_feats2)

        out_edge_feats = out_edge_feats[:, 0, :]
        return out_edge_feats#self.pred(g, h, h2)


class DistanceModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.egatcon = EGATConv(in_node_feats=in_features,
                                in_edge_feats=in_features,
                                out_node_feats=out_features,
                                out_edge_feats=out_features,
                                num_heads=1)
        self.pred = MLPPredictor(out_features, 3)#

    def forward(self, g, x, e):
        h = self.sage(g, x)
        new_node_feats, h2 = self.egatcon(g, x, e)
        h2 = h2[:, 0, :]
        return self.pred(g, h, h2)