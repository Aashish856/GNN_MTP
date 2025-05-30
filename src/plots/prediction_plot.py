import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
import plot_config
from sklearn.metrics import r2_score, mean_squared_error

def prediction_plot(x, y, x_labels, y_labels, common_title):
    """
    Plots 4 scatter plots in a row comparing predictions vs true values, with R^2 and MSE.

    Parameters:
    - x: numpy array of shape (n, 4) for true values
    - y: numpy array of shape (n, 4) for predicted values
    - x_labels: list of 4 strings for x-axis labels
    - y_labels: list of 4 strings for y-axis labels
    - common_title: string for the common title of the figure
    """
    # Create the directory if it doesn't exist
    save_dir = "figures/prediction_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Create file name
    file_name = f"{common_title}".replace(" ", "_").replace("/", "_") + ".png"
    save_path = os.path.join(save_dir, file_name)

    # Set up the 1x4 subplot grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        ax = axes[i]
        ax.scatter(x[:, i], y[:, i], color="#0000FF", alpha=0.7, s=5)

        # Add y=x line
        # min_val = min(np.min(x[:, i]), np.min(y[:, i]))
        # max_val = max(np.max(x[:, i]), np.max(y[:, i]))
        # ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        # Compute R^2 score and MSE
        r2 = r2_score(x[:, i], y[:, i])
        mse = mean_squared_error(x[:, i], y[:, i])

        # Labels
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel(y_labels[i])

        # Write R^2 and MSE text
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.4f}\nMSE = {mse:.4f}", transform=ax.transAxes,
                fontsize=11, verticalalignment='top', color="black")

        # Remove gridlines
        ax.grid(False)

    # Common title
    fig.suptitle(common_title, fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and close
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")