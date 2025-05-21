import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_compression(embeddings, compressed_dim=2, perplexity=100, learning_rate=200, n_iter=1000):
    """
    Perform t-SNE compression on embeddings.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (num_samples, embedding_dim)
        compressed_dim (int): Target dimension for t-SNE
        perplexity (float): Perplexity parameter for t-SNE (suggested 5-50)
        learning_rate (float): Learning rate for t-SNE optimization
        n_iter (int): Number of optimization iterations

    Returns:
        compressed_embedding (np.ndarray): Embedding reduced to compressed_dim dimensions
    """

    tsne = TSNE(
        n_components=compressed_dim,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
        init='random',
        verbose=1
    )

    compressed_embedding = tsne.fit_transform(embeddings)

    return compressed_embedding