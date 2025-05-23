import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import trustworthiness

def umap_compression_analysis(embeddings, target_dim, n_neighbors=100, min_dist=0.1):
    """
    Perform UMAP compression to target_dim and plot trustworthiness for dimensions 2 to 10.

    Args:
        embeddings (np.ndarray): Original high-dimensional embeddings (N x D).
        target_dim (int): Desired compressed dimension to return.

    Returns:
        compressed_embeddings (np.ndarray): Embeddings compressed to target_dim using UMAP.
    """

    trust_scores = []
    dims = list(range(2, 11))

    for dim in dims:
        reducer = umap.UMAP(n_components=dim, n_neighbors = n_neighbors, min_dist = min_dist,  random_state=42)
        embedding_dim = reducer.fit_transform(embeddings)
        score = trustworthiness(embeddings, embedding_dim, n_neighbors=n_neighbors)
        trust_scores.append(score)

    # Plot trustworthiness vs. dimension
    plt.figure(figsize=(8, 5))
    plt.plot(dims, trust_scores, marker='o')
    plt.xlabel('Compressed Dimension')
    plt.ylabel('Trustworthiness Score')
    plt.title('UMAP Trustworthiness Analysis')
    plt.grid(True)
    plt.show()

    # Final compression to target_dim
    reducer = umap.UMAP(n_components=target_dim, n_neighbors = n_neighbors, min_dist = min_dist, random_state=42)
    compressed_embeddings = reducer.fit_transform(embeddings)

    return compressed_embeddings
