import torch
import torch.nn as nn
import torch.optim as optim

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MessageParsingLayer(nn.Module):
    def __init__(self, h_dim):
        super(MessageParsingLayer, self).__init__()
        self.edge_mlp = MLP(h_dim, h_dim, h_dim)
        self.agg_mlp = MLP(2*h_dim, h_dim, h_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        source = x[row]
        target = x[col]
        out = source - target
        e = self.edge_mlp(out)
        agg = unsorted_segment_sum(e, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        return self.agg_mlp(agg)

class GNN(nn.Module):
    def __init__(self, n_layers, h_dim, n_atm, num_cvs=4):
        super(GNN, self).__init__()
        self.n_layers = n_layers
        self.n_atm = n_atm
        self.h_dim = h_dim
        self.num_cvs = num_cvs

        self.embedding = nn.Linear(3, h_dim)

        for i in range(n_layers):
            self.add_module(f"mpl_{i}", MessageParsingLayer(h_dim))

        self.node_decoding = MLP(h_dim, h_dim, h_dim)

        self.cv_decoders = nn.ModuleList([MLP(h_dim, h_dim, 1) for _ in range(num_cvs)])

    def forward(self, x, edge_index):
        h = self.embedding(x)
        for i in range(self.n_layers):
            h = self._modules[f"mpl_{i}"](h, edge_index)

        h = self.node_decoding(h)
        h = h.view(-1, self.n_atm, self.h_dim)
        h = torch.mean(h, dim=1)

        outputs = [decoder(h) for decoder in self.cv_decoders]
        return torch.cat(outputs, dim=1)

def gnn_model(h_dim, n_layers, n_atm, starting_learning_rate, device):
  model = GNN(h_dim=h_dim, n_layers=n_layers, n_atm = n_atm).to(device)
  optimizer = optim.Adam(model.parameters(), lr=starting_learning_rate)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999924)
  return model, optimizer, scheduler        
