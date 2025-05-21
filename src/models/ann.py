def normalization(embeddings):
    """
    Normalize the embeddings to have zero mean and unit variance.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (num_samples, embedding_dim)

    Returns:
        np.ndarray: Normalized embeddings
    """
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    normalized_embeddings = (embeddings - mean) / std
    return normalized_embeddings, mean, std

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
