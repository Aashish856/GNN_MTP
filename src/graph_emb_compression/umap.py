import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import trustworthiness

def umap_compression_analysis(embeddings, target_dim):
    """
    Perform UMAP compression to target_dim and plot trustworthiness for dimensions 2 to 10.

    Args:
        embeddings (np.ndarray): Original high-dimensional embeddings (N x D).
        target_dim (int): Desired compressed dimension to return.

    Returns:
        compressed_embeddings (np.ndarray): Embeddings compressed to target_dim using UMAP.
    """

    trust_scores = []
    dims = list(range(1, 11))

    for dim in dims:
        reducer = umap.UMAP(n_components=dim,  random_state=42)
        embedding_dim = reducer.fit_transform(embeddings)
        score = trustworthiness(embeddings, embedding_dim)
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
    reducer = umap.UMAP(n_components=target_dim, random_state=42)
    compressed_embeddings = reducer.fit_transform(embeddings)

    return np.array(trust_scores), compressed_embeddings
