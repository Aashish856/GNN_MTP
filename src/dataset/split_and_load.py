from sklearn.model_selection import train_test_split
from src.utils.helper import getDataLoader
import numpy as np


def prepare_data_loaders_from_dataset(dataset, test_size=0.1, batch_size=16, random_state=69):
    """
    Prepare train and validation data loaders directly from a CVDataset object.

    Args:
        dataset (CVDataset): Your dataset instance containing coords and CVs.
        test_size (float): Fraction of data to use as validation set.
        batch_size (int): Batch size for DataLoader.
        random_state (int): Seed for reproducibility.

    Returns:
        train_loader, val_loader, train_targets, val_targets
    """

    # Get inputs and all CVs as stacked targets
    inputs = dataset.get_coordinates()  # e.g. shape (N, nodes, 3)
    all_cvs = dataset.get_all_cvs()    # dict: cv_name -> array of shape (N,) or (N, d)

    # Stack all CVs along axis=1, so targets shape is (N, num_cvs)
    target_list = [np.array(cv) for cv in all_cvs.values()]
    targets = np.stack(target_list, axis=1)

    # Split train and val
    inp_train, inp_val, targets_train, targets_val = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )

    # Create DataLoaders using your utility
    train_loader = getDataLoader(inp_train, targets_train, batch_size=batch_size)
    val_loader = getDataLoader(inp_val, targets_val, batch_size=batch_size)

    return train_loader, val_loader, targets_train, targets_val
