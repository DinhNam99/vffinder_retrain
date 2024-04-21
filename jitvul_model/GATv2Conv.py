import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import LeakyReLU, ELU
from torch_geometric.nn import GCNConv, GATv2Conv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out, dropout = 0.1,):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)
    self.relu = LeakyReLU(inplace=True)
    self.lin = Linear(dim_out, 2)
    self.dropout = dropout

  def forward(self, x, edge_index):
    # x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.gcn1(x, edge_index)
    x = self.relu(x)
    # x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.gcn2(x, edge_index)
    x = global_add_pool(x,  torch.zeros(
                x.shape[0], dtype=int, device=x.device))
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.lin(x)
    return x


class GATv2(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out ,dropout = 0.1, heads=8,):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads, dropout=dropout)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1, dropout=dropout)
    self.relu = LeakyReLU(inplace=True)
    self.dropout = dropout
    self.lin = Linear(dim_out, 2)

  def forward(self, x, edge_index, edge_type, edge_attr):
    # x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.gat1(x, edge_index)
    x = self.relu(x)
    # x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.gat2(x, edge_index)
    x = global_add_pool(x,  torch.zeros(
                x.shape[0], dtype=int, device=x.device))
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.lin(x)
    return x

class GATv1(torch.nn.Module):
  def __init__(self, dim_in, dim_h, dim_out ,dropout = 0.1, heads=8,):
    super().__init__()  
    self.conv1 = GATConv(dim_in, dim_h, heads=heads, dropout=dropout)
    self.conv2 = GATConv(dim_h*heads, dim_out, heads=1, concat=False, dropout=dropout)
    self.relu = LeakyReLU(inplace=True)
    self.dropout = dropout
    self.lin = Linear(dim_out, 2)
  def forward(self, x, edge_index, edge_type, edge_attr):
    # x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.conv1(x, edge_index)
    x = self.relu(x)
    # x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.conv2(x, edge_index)
    x = global_add_pool(x,  torch.zeros(
                x.shape[0], dtype=int, device=x.device))
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.lin(x)
    return x