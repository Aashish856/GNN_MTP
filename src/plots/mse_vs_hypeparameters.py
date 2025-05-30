import matplotlib.pyplot as plt
import numpy as np
import os

def mse_vs_hypeparameters(x, y, cv_labels, title):
    """
    Plots MSE vs Hidden Dimension Size for multiple CVs.

    Parameters:
    - mse_values: numpy array of shape (4, 4) [CVs x hidden sizes]
    - hidden_dims: list of 4 hidden dimension sizes [8, 16, 32, 64]
    - cv_labels: list of 4 strings for CV names
    - title: string for plot title
    """

    # Create directory if needed
    save_dir = "figures/mse_vs_hidden_dim"
    os.makedirs(save_dir, exist_ok=True)

    # File name
    file_name = f"{title}".replace(" ", "_").replace("/", "_") + ".png"
    save_path = os.path.join(save_dir, file_name)

    # Set up figure
    plt.figure(figsize=(8, 6))

    # Colors (4 distinct colors)
    colors = ["#0000FF", "#FF0000", "#00C800", "#FFA500"]

    for i in range(4):
        plt.plot(hidden_dims, y[i], marker='o', color=colors[i], label=cv_labels[i], linewidth=2)

    # Labels, title, legend
    plt.xlabel("Hidden Dimension Size", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()

    plt.show()
    # Save and show
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")