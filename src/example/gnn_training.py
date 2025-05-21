# # You can run this script directly or in a Jupyter notebook/Google Collab.

# import os
# def get_base_dir():
#     try:
#         # When running from a .py script
#         return os.path.dirname(os.path.abspath(__file__))
#     except NameError:
#         # When running interactively in a notebook
#         return os.getcwd()

# BASE_DIR = get_base_dir()
# print(BASE_DIR)

# !git clone https://github.com/Aashish856/GNN_MTP.git # Cloning the repository

# !pip install -r ./GNN_MTP/requirements.txt # Installing the requirements

# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import gdown
# import pickle

# from GNN_MTP.src.train import train, run_training
# from GNN_MTP.src.dataset.split_and_load import prepare_data_loaders_from_dataset
# from GNN_MTP.src.dataset.data import CVDataset


# # Downloading Dataset from google drive and saving it in the dataset directory

# DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
# os.makedirs(DATASET_DIR, exist_ok=True)

# files_to_download = {
#     "1eTNfXgmK7Okmvl00j6TSFga_nbyVVVkg" : "final_coordinates.pkl",
#     "1vtYy-GRXDSRfC4THdxUj7bJ6apghEuI9": "s_list.pkl",
#     "1YefRlv5fI0wnS4PXC_R7QFzSqSJA5ng0": "xi_list.pkl",
#     "1eWNd7zH9L3ItX0rQU0lccbkBhaTTOjXJ": "xi_mf_list.pkl",
#     "1YmuJAsfImqzB5RyCpAgYZQKVWe46o1Sn": "z_list.pkl",
#     "1eTNfXgmK7Okmvl00j6TSFga_nbyVVVkg" : "train_test_indices.pkl",
# }

# # Download files
# for file_id, file_name in files_to_download.items():
#     output_path = os.path.join(DATASET_DIR, file_name)
#     url = f"https://drive.google.com/uc?id={file_id}"
#     print(f"Downloading {file_name}...")
#     gdown.download(url, output_path, quiet=False)

# print("All files downloaded successfully to:", DATASET_DIR)

# # Defining Dataset File Path
# coord_path = os.path.join(BASE_DIR, 'dataset', "final_coordinates.pkl")
# cv_paths = {
#     's': os.path.join(BASE_DIR, 'dataset', "s_list.pkl"),
#     'xi': os.path.join(BASE_DIR, 'dataset', "xi_list.pkl"),
#     'xi_mf': os.path.join(BASE_DIR, 'dataset', "xi_mf_list.pkl"),
#     'z': os.path.join(BASE_DIR, 'dataset', "z_list.pkl")   
# }

# # Creating Dataset Object

# dataset = CVDataset(coord_path, cv_paths)

# # Defining Hyperparameters
# batch_size = 16
# learning_rate = 0.0008
# test_size = 0.11
# h_dims = 16
# cutoff = 0.26
# n_layers = 3
# n_atm = 3072
# device = "cuda"
# num_epochs = 20
# perform_rotations = True
# loss_fn = nn.MSELoss()

# # Preparing Data Loaders

# # Check if train_test_indices.pkl exists
# indices_path = os.path.join(DATASET_DIR, "train_test_indices.pkl")
# if os.path.exists(indices_path):
#     with open(indices_path, "rb") as f:
#         indices = pickle.load(f)
#     train_indices = indices["train_indices"]
#     test_indices = indices["test_indices"]

#     prepare_data_loaders_from_dataset(dataset, test_size=test_size, batch_size=batch_size, train_indices=train_indices, test_indices=test_indices)    
# else:
#     # If not, create new indices
#     print("train_test_indices.pkl not found. Creating new train/test indices. and dataloaders.")
#     train_loader, val_loader, targets_train, targets_val, train_indices, test_indices = prepare_data_loaders_from_dataset(dataset, test_size=test_size, batch_size=batch_size)
#     indices_dir = "results"
#     os.makedirs(indices_dir, exist_ok=True)
#     indices_path = os.path.join(indices_dir, "train_test_indices.pkl")
#     with open(indices_path, "wb") as f:
#         pickle.dump({"train_indices": train_indices, "test_indices": test_indices}, f)
#     print(f"Saved train and test indices to {indices_path}")

# del dataset # Freeing up memory

# run_training(
#     h_dim=h_dim,
#     cutoff=cutoff,
#     n_layer=n_layers,
#     n_atm = n_atm,
#     train_dataloader=train_loader,
#     val_dataloader=val_loader,
#     loss_fn=loss_fn,
#     device=device,
#     num_epochs=num_epochs,
#     perform_rotations = perform_rotations
# )

