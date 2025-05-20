from sklearn.model_selection import train_test_split
from ..utils.helper import getDataLoader
import numpy as np


def prepare_data_loaders_from_dataset(
    dataset, test_size=0.1, batch_size=16, random_state=69,
    train_indices=None, test_indices=None
):
    """
    Prepare train and validation data loaders directly from a CVDataset object.

    Args:
        dataset (CVDataset): Your dataset instance containing coords and CVs.
        test_size (float): Fraction of data to use as validation set (ignored if indices provided).
        batch_size (int): Batch size for DataLoader.
        random_state (int): Seed for reproducibility.
        train_indices (array-like, optional): Indices for training set.
        test_indices (array-like, optional): Indices for validation set.

    Returns:
        train_loader, val_loader, targets_train, targets_val, train_indices, test_indices
    """

    # Get inputs and all CVs as stacked targets
    inputs = dataset.get_coordinates()  # shape (N, nodes, 3)
    all_cvs = dataset.get_all_cvs()    # dict: cv_name -> array of shape (N,) or (N, d)

    # Stack all CVs along axis=1, so targets shape is (N, num_cvs)
    target_list = [np.array(cv) for cv in all_cvs.values()]
    targets = np.stack(target_list, axis=1)

    num_samples = inputs.shape[0]

    if train_indices is None or test_indices is None:
        # Generate new train/test split
        train_indices, test_indices = train_test_split(
            np.arange(num_samples),
            test_size=test_size,
            random_state=random_state
        )

    # Subset the data using indices
    inp_train, inp_val = inputs[train_indices], inputs[test_indices]
    targets_train, targets_val = targets[train_indices], targets[test_indices]

    # Create DataLoaders
    train_loader = getDataLoader(inp_train, targets_train, batch_size=batch_size)
    val_loader = getDataLoader(inp_val, targets_val, batch_size=batch_size)

    return train_loader, val_loader, targets_train, targets_val, train_indices, test_indices