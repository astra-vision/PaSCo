import torch
import torch.nn as nn
from pykeops import set_verbose
from pykeops.torch import Vi, Vj

set_verbose(False)


class knn_up(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, v_coor, v_feats, p_coor):
        """
        Input:
            v_coor: vox points coords [M, C]
            v_feats: vox feats [M, D]
            p_coor: points coords [N, C]
        Return:
            interp_feat: point_feats [N, D], N>M
        """
        N, _ = p_coor.shape
        dists, idx = kNN(v_coor, p_coor, self.k)
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm
        interp_feat = torch.sum(
            index_points(v_feats, idx) * weight.view(N, self.k, 1), dim=1
        )
        return interp_feat


def kNN(x_train, x_test, K):
    # Encoding as KeOps LazyTensors:
    D = x_train.shape[-1]
    X_i = Vi(0, D)  # Purely symbolic "i" variable, without any data array
    X_j = Vj(1, D)  # Purely symbolic "j" variable, without any data array
    # Symbolic distance matrix:
    D_ij = ((X_i - X_j) ** 2).sum(-1)
    # K-NN query operator:
    KNN = D_ij.Kmin_argKmin(K, dim=1)
    # Actual K-NN query:
    values, indices = KNN(x_test, x_train)
    return values, indices


def index_points(points, idx):
    """
    Input:
        points: input point features [N, C]
        idx: sample index data [M]
    Return:
        new_points:, indexed features [M, C]
    """
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    new_points = points[idx, :]
    return new_points


if __name__ == "__main__":
    import open3d as o3d

    up = knn_up(4)

    # Create voxel points
    vx, vy, vz = torch.meshgrid(torch.arange(2), torch.arange(2), torch.arange(2))
    vox_coors = torch.stack((vx, vy, vz), dim=-1).reshape(2 * 2 * 2, 3).float()
    vox_coors[:, 0] = vox_coors[:, 0] + 0.5
    vox_coors[:, 1] = vox_coors[:, 1] + 0.5
    vox_coors[:, 2] = vox_coors[:, 2] + 0.5

    # Vox colors
    color_vox = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0.5, 0.5, 0.5],
            [0, 0, 0],
        ]
    )
    pcd_vox = o3d.geometry.PointCloud()
    pcd_vox.points = o3d.utility.Vector3dVector(vox_coors.numpy())
    pcd_vox.colors = o3d.utility.Vector3dVector(color_vox)

    # Create points
    px, py, pz = torch.meshgrid(
        torch.arange(20) / 10, torch.arange(20) / 10, torch.arange(20) / 10
    )
    pts_coors = torch.stack((px, py, pz), dim=-1).reshape(20 * 20 * 20, 3)
    pcd_pts = o3d.geometry.PointCloud()
    pcd_pts.points = o3d.utility.Vector3dVector(pts_coors)

    # Show voxel colors and points
    pcd_pts.paint_uniform_color([0.1, 0.1, 0.1])
    o3d.visualization.draw_geometries([pcd_pts, pcd_vox])

    # Get points colors through interpolation
    color_pts = up(vox_coors, color_vox, pts_coors)
    pcd_pts.colors = o3d.utility.Vector3dVector(color_pts)
    o3d.visualization.draw_geometries([pcd_pts, pcd_vox])
