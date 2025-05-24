class ANN(nn.Module):
    def __init__(self, embedding_dim=32, num_cvs=4):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim),
            nn.ReLU(),
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_cvs)
        )
    def forward(self, x):
        return self.model(x)


def ann_model(emb_dim, num_cvs, device, starting_learning_rate=0.001):
  model = ANN(embedding_dim=emb_dim, num_cvs=num_cvs).to(device)  
  optimizer = optim.Adam(model.parameters(), lr=starting_learning_rate)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999924)
  return model, optimizer, scheduler        
