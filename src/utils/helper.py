import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def normalize_coordinates(coords: np.ndarray):
    """
    Normalize coordinates array by mean-centering and scaling.

    Args:
        coords (np.ndarray): Coordinates of shape (N_samples, N_features, ...)

    Returns:
        normalized_coords (np.ndarray): Normalized coordinates
        means (np.ndarray): Means used for centering
        min_coords (np.ndarray): Minimum values after centering
        max_coords (np.ndarray): Maximum values after centering and scaling
    """
    means = np.mean(coords, axis=1, keepdims=True)
    stds = np.std(coords, axis=1, keepdims=True)
    coords_centered = coords - means
    min_coords = np.min(coords_centered, axis=(0,1), keepdims=True)
    max_coords = np.max(coords_centered, axis=(0,1), keepdims=True)
    scale_factor = np.std(coords_centered)
    normalized_coords = coords_centered / scale_factor

    return normalized_coords

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def getDataLoader(x,y, batch_size=16):
    tensor_inp = torch.Tensor(x)
    tensor_z = torch.Tensor(y)
    dataset = TensorDataset(tensor_inp,tensor_z)
    return DataLoader(dataset, batch_size)

def pairwise_distances(x):
    square = torch.sum(x ** 2, dim=2, keepdim=True)
    distances = square + torch.transpose(square, 1, 2) - 2 * torch.matmul(x, torch.transpose(x, 1, 2))
    distances = torch.abs(distances)
    distances = torch.sqrt(distances)
    return distances
