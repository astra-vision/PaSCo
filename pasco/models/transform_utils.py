import torch
import numpy as np
from scipy.spatial.transform import Rotation
import torch.nn.functional as F


def generate_transformation(
    rot=0.0, translation=(0.0, 0.0, 0.0), flip_dim=None, scale=1  # in meters
):
    # * flip
    T_flip = np.identity(4)
    if flip_dim is not None:
        T_flip[flip_dim, flip_dim] = -1
    T_flip = torch.from_numpy(T_flip).float()

    # * rotation and translation
    r = Rotation.from_euler("xyz", [0, 0, rot], degrees=True)
    T = np.identity(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = translation

    T = torch.from_numpy(T).float()

    # * scale
    T_scale = np.identity(4)
    T_scale[[0, 1, 2], [0, 1, 2]] *= scale
    T_scale = torch.from_numpy(T_scale).float()

    return T_scale @ T @ T_flip


def generate_random_transformation(
    max_angle=45, flip=True, scale_range=0.1, max_translation=np.array([1.0, 1.0, 0.5])
):
    translation = (np.random.rand(3) - 0.5) * max_translation
    rot = (np.random.rand() - 0.5) * max_angle * 2
    if flip and np.random.rand() > 0.5:
        flip_dim = 1
    else:
        flip_dim = None
    scale = 1.0 + (np.random.rand(3) - 0.5) * scale_range

    # print(rot, translation, flip_dim, scale)
    return generate_transformation(
        rot=rot, translation=translation, flip_dim=flip_dim, scale=scale
    )


def transform_xyz(points, T):
    points = points.clone()

    homogenous_points = torch.cat(
        [points, torch.ones(points.shape[0], 1).to(points.device)], dim=1
    ).type_as(T)
    new_points = (T @ homogenous_points.T).T[:, :3]

    return new_points


def transform(coords, T, resolution=0.2):
    min_bound = torch.tensor([0, -25.6, -2]).reshape(1, 3).to(coords.device)
    max_bound = torch.tensor([51.2, 25.6, 4]).reshape(1, 3).to(coords.device)
    points = coords * resolution + resolution / 2

    points = min_bound + points

    homogenous_points = torch.cat(
        [points, torch.ones(points.shape[0], 1).to(points.device)], dim=1
    ).type_as(T)
    new_points = (T @ homogenous_points.T).T[:, :3]
    new_points = (new_points - min_bound - resolution / 2) / resolution
    new_coords = torch.round(new_points).int()

    return new_coords


def sample_grid_coords(dims):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])

    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(float)

    return coords_grid, g_xx, g_yy, g_zz


def sample_grid_features(coords, voxels):
    """
    coords: B, 3 # the 2 columns store x, y, z coords
    voxels: F, H, W, D
    -------------
    return
    color_bilinear: 3, B
    """
    _, H, W, D = voxels.shape
    coords = coords.float()
    coords_t = torch.ones_like(coords)  # B, 3
    coords_t[:, 0] = (coords[:, 2] / (D - 1) - 0.5) * 2
    coords_t[:, 1] = (coords[:, 1] / (W - 1) - 0.5) * 2
    coords_t[:, 2] = (coords[:, 0] / (H - 1) - 0.5) * 2

    sampled_features = F.grid_sample(
        voxels.unsqueeze(0).float(),
        coords_t.unsqueeze(0).unsqueeze(0).unsqueeze(0).float(),
        align_corners=True,
        mode="nearest",
        padding_mode="zeros",
    )
    return sampled_features.reshape(-1, coords_t.shape[0])


def transform_scene(from_coords, T, voxel_features, to_coords_bnd=None):
    """
    Only implement for batch size = 1
    Steps to transform a grid to avoid holes:
    1. Determine the new grid size
    2. Sample the new grid
    3. Project the new grid back to the original grid
    4. Sample the original grid features at the projected grid
    Args:
        coords: (N, 3) tensor of discrete coordinates
        T: (4, 4) transformation matrix
        projected_voxel_features: (F, H, W, D) tensor of voxel features (H, W, D) are spatial dimensions
    Returns:
        sampled_features: (N, F) tensor of sampled features
        new_coords: (N, 3) tensor of new coordinates
    """
    # * 1. Determine the new grid size
    if to_coords_bnd is None:
        to_coords = transform(from_coords, T)
        min_to_coords, max_to_coords = to_coords.min(0)[0], to_coords.max(0)[0]
        to_coords_bnd = (min_to_coords, max_to_coords)
    else:
        min_to_coords, max_to_coords = to_coords_bnd
    to_grid_size = max_to_coords - min_to_coords + 1
    to_grid_size = to_grid_size.cpu().numpy()

    # * 2. Sample the new grid
    to_coords = sample_grid_coords(to_grid_size)[0]
    to_coords = torch.from_numpy(to_coords).type_as(from_coords)
    to_coords = to_coords + min_to_coords.reshape(1, 3)

    # * 3. Project the new grid back to the original grid
    to_coords_projected = transform(to_coords, torch.inverse(T))

    # * 4. Sample the original grid features at the projected grid
    sampled_features = sample_grid_features(to_coords_projected, voxel_features)

    return sampled_features.T, to_coords.int(), to_coords_bnd


def sample_scene(
    min_to_coords: int, T, to_voxel_features, out_scene_size, resolution=0.2
):
    """
    Only implement for batch size = 1
    Steps to transform a grid to avoid holes:
    2. Sample the new grid
    3. Project the new grid back to the original grid
    4. Sample the original grid features at the projected grid
    """
    # * 1. Sample the new grid
    from_coords = sample_grid_coords(out_scene_size)[0]
    from_coords = torch.from_numpy(from_coords).to(min_to_coords.device)

    # * 2. Project the new grid back to the original grid
    from_coords_projected = transform(from_coords, T, resolution=resolution)

    # * 3. Sample the original grid features at the projected grid
    from_coords_projected = from_coords_projected - min_to_coords
    sampled_features = sample_grid_features(from_coords_projected, to_voxel_features)

    return sampled_features.T, from_coords
