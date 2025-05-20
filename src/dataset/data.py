# src/data/dataset.py
import pickle
import numpy as np
from src.utils.helper import normalize_coordinates

class CVDataset:
    """
    Dataset class for loading coordinates and multiple CVs from pickle files.
    """
    def __init__(self, coord_path, cv_paths):
        """
        Initialize the dataset.

        Args:
            coord_path (str): Path to pickle file containing coordinates.
            cv_paths (dict): Dictionary with keys as CV names and values as file paths.
                             Example: {'s': 's_list.pkl', 'xi': 'xi_list.pkl', ...}
        """
        self.coordinates = normalize_coordinates(self.load_pickle(coord_path))
        self.cvs = {name: self.load_pickle(path) for name, path in cv_paths.items()}

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def get_coordinates(self):
        return self.coordinates

    def get_cv(self, name):
        return self.cvs.get(name, None)

    def get_all_cvs(self):
        return self.cvs
