import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_compression_analysis(embeddings, accepted_variance=0.95):
    """
    Perform PCA compression analysis on embeddings and plot cumulative explained variance.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (num_samples, embedding_dim)
        accepted_variance (float): Target cumulative variance to retain (default 0.95)

    Returns:
        n_components_optimal (int): Number of components needed to retain accepted_variance
        compressed_embeddings (np.ndarray): Transformed embeddings with reduced dimensions
    """

    # Fit PCA
    pca = PCA()
    pca.fit(embeddings)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Compression Analysis')
    plt.grid(True)
    plt.show()

    # Find number of components to retain accepted_variance
    n_components_optimal = np.argmax(cumulative_variance >= accepted_variance) + 1
    print(f"Number of components to retain {accepted_variance*100:.1f}% variance: {n_components_optimal}")

    # Transform embeddings to compressed space
    pca = PCA(n_components=n_components_optimal)
    compressed_embeddings = pca.fit_transform(embeddings)

    return n_components_optimal, compressed_embeddings
