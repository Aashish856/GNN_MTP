import matplotlib.pyplot as plt
import numpy as np
import os
import plot_config  # assuming your config file is named plot_config.py

def prediction_plot(x, y, x_label, y_label, title):
    # Create the directory if it doesn't exist
    save_dir = "figures/prediction_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Create file name by replacing spaces with underscores and removing special characters
    file_name = f"{y_label}_{title}".replace(" ", "_").replace("/", "_") + ".png"
    save_path = os.path.join(save_dir, file_name)

    # Create plot
    plt.figure()
    plt.scatter(x, y, color=plot_config.PRIMARY_COLOR, alpha=0.7)

    # # Add y=x line
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save and close
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")