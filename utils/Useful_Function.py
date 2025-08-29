import torch
import numpy as np
def FT(variable):
    return torch.FloatTensor(variable).to(torch.device("cuda:0"))

def create_square_positions(num_agents, initial_height=0, spacing=1.0, center=True):
    """
    Create positions arranged in a square grid optimized for GPU pipeline.

    Args:
        num_agents: Total number of agents (must be a perfect square for exact square)
        initial_height: Initial z-coordinate for all agents
        spacing: Distance between agents
        center: If True, centers the grid around (0,0,0)
        device: torch.device ('cuda', 'cpu', or None for auto-detection)

    Returns:
        torch.Tensor: Positions tensor of shape (num_agents, 3) on specified device
        torch.Tensor: Orientations tensor of shape (num_agents, 4) on specified device
    """
    device = torch.device("cuda:0")
    # Calculate grid size (keeping computation on CPU for scalar operations)
    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(num_agents))))

    # Create grid coordinates directly on target device
    x = torch.linspace(0, (grid_size - 1) * spacing, grid_size, device=device)
    y = torch.linspace(0, (grid_size - 1) * spacing, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Create positions tensor in one go
    positions = torch.empty((num_agents, 3), device=device)
    positions[:, 0] = xx.flatten()[:num_agents]
    positions[:, 1] = yy.flatten()[:num_agents]
    positions[:, 2] = initial_height

    if center:
        # Center the grid - single operation on GPU
        offset = (grid_size - 1) * spacing / 2
        positions[:, :2] -= offset

    # Create orientation quaternions (identity quaternions)
    initial_ori = torch.zeros((num_agents, 4), device=device)
    initial_ori[:, 0] = 1  # w component of quaternion

    return positions, initial_ori

def get_euler_angle(quat): # I have checked it, it is correct
    # Input quat shape: (n, 4) in w,x,y,z order
    # Output shape: (n, 3) in x, y, z but derived from ZYX Euler angle

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    # Roll (x), Pitch (y), Yaw (z)
    # Using the ZYX convention XYZ Euler

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Use 1.0 - 1e-6 to avoid NaN when |sinp| is slightly > 1.0 due to floating point
    sinp = torch.clamp(sinp, -1.0 + 1e-6, 1.0 - 1e-6)
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    angles = torch.stack([roll, pitch, yaw], dim=1)

    # Angle adjustments to keep them within [-pi/2, pi/2]
    angles = torch.where(angles < -np.pi / 2, angles + np.pi, angles)
    angles = torch.where(angles > np.pi / 2, angles - np.pi, angles)

    return angles


def State_Normalization(state):
    return state

