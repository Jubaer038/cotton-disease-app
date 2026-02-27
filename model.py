import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class CLIP_GCN_LearnableAdj(nn.Module):
    def __init__(self, num_classes, clip_feature_dim=512, gcn_hidden=256, dropout=0.3):
        super(CLIP_GCN_LearnableAdj, self).__init__()

        self.adj_learner = nn.Linear(clip_feature_dim, clip_feature_dim)

        self.gcn1 = GCNConv(clip_feature_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)

        self.bn1 = nn.BatchNorm1d(gcn_hidden)
        self.bn2 = nn.BatchNorm1d(gcn_hidden)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gcn_hidden, num_classes)

    def learn_adjacency(self, x):
        transformed = self.adj_learner(x)
        similarity = torch.mm(transformed, transformed.t())
        attention = F.softmax(similarity, dim=-1)
        return attention

    def forward(self, x):
        learned_adj = self.learn_adjacency(x)
        edge_index, edge_weight = dense_to_sparse(learned_adj)

        x = self.gcn1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gcn2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.classifier(x)
