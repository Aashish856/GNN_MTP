import torch

# Rotation matrix around z-axis by angle theta
def rotation_matrix_z(theta):
    theta = torch.tensor(theta, dtype=torch.float32)  # Convert to tensor
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta),  torch.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

# Rotation matrix around y-axis by angle phi
def rotation_matrix_y(phi):
    phi = torch.tensor(phi, dtype=torch.float32)  # Convert to tensor
    return torch.tensor([
        [torch.cos(phi), 0.0, torch.sin(phi)],
        [0.0, 1.0, 0.0],
        [-torch.sin(phi), 0.0, torch.cos(phi)]
    ], dtype=torch.float32)

# Rotation matrix around x-axis by angle gamma
def rotation_matrix_x(gamma):
    gamma = torch.tensor(gamma, dtype=torch.float32)  # Convert to tensor
    return torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, torch.cos(gamma), -torch.sin(gamma)],
        [0.0, torch.sin(gamma),  torch.cos(gamma)]
    ], dtype=torch.float32)

# Apply rotation to a frame by angles theta, phi, and gamma
def rotate_frame(frame, theta, phi, gamma):
    Rz = rotation_matrix_z(theta).to(frame.device)  # Rotation around z-axis
    Ry = rotation_matrix_y(phi).to(frame.device)    # Rotation around y-axis
    Rx = rotation_matrix_x(gamma).to(frame.device)  # Rotation around x-axis

    # Combined rotation matrix: R = Rz * Ry * Rx (rotation order matters)
    R = torch.matmul(Rz, torch.matmul(Ry, Rx))  # R = Rz * Ry * Rx
    return torch.matmul(frame, R.T)  # Apply rotation to frame

# Randomly rotate all frames in tensor x
def randomRotate(x):
    # x.shape = (batch_size, 3072, 3)
    frames = x
    batch_size, num_nodes, _ = frames.shape

    # Create an empty tensor for rotated frames
    rotated_frames = torch.zeros_like(frames)

    # Generate random angles and rotate each frame
    for i in range(batch_size):
        theta_deg = torch.rand(1).item() * 360  # Random angle in degrees (Z-axis)
        phi_deg = torch.rand(1).item() * 360    # Random angle in degrees (Y-axis)
        gamma_deg = torch.rand(1).item() * 360  # Random angle in degrees (X-axis)

        # Convert degrees to radians manually
        theta_rad = theta_deg * (torch.pi / 180.0)
        phi_rad = phi_deg * (torch.pi / 180.0)
        gamma_rad = gamma_deg * (torch.pi / 180.0)

        # Rotate the frame
        rotated_frames[i] = rotate_frame(frames[i], theta_rad, phi_rad, gamma_rad)

    return rotated_frames