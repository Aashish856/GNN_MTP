import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rcParams

def plot_graph_emb(graph_emb, n_bins=100, save_path="graph_embs.png"):
    """
    Plots graph embedding dimension variations over simulation,
    sorted by correlation with simulation progress (y=x).

    Parameters:
    - graph_emb (ndarray): (n_frames, n_dimensions) embedding array.
    - n_bins (int): Number of bins to divide frames (default 100).
    - save_path (str): Path to save figure (supports .png, .pdf).
    """
    n_points, n_dims = graph_emb.shape
    bin_size = n_points // n_bins

    sim_progress = np.arange(n_points)

    # Compute correlation with y=x (simulation progress) for each dimension
    corrs = np.array([np.corrcoef(sim_progress, graph_emb[:, i])[0, 1] for i in range(n_dims)])

    # Sort dimensions by correlation (descending)
    sorted_indices = np.argsort(-corrs)

    n_cols = 4
    n_rows = math.ceil(n_dims / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), dpi=300)
    axes = axes.flatten()

    for plot_idx, dim_idx in enumerate(sorted_indices):
        binned_data = [graph_emb[j * bin_size:(j + 1) * bin_size, dim_idx] for j in range(n_bins)]
        bin_means = np.array([np.mean(bin_data) for bin_data in binned_data])
        bin_mins  = np.array([np.min(bin_data)  for bin_data in binned_data])
        bin_maxs  = np.array([np.max(bin_data)  for bin_data in binned_data])
        ax = axes[plot_idx]
        ax.plot(range(n_bins), bin_means, color=PRIMARY_COLOR, label='Mean')
        ax.fill_between(range(n_bins), bin_mins, bin_maxs, color=PRIMARY_COLOR_LIGHT, alpha=0.5, label='Min-Max')
        ax.set_title(f'Dim {plot_idx}', fontsize=16)
        ax.set_xlabel('Simulation Progress', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.grid(True)
        if plot_idx == 0:
            ax.legend()
    # Remove unused subplots
    for j in range(n_dims, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)