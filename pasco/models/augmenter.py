import MinkowskiEngine as ME
import torch
import numpy as np


class Augmenter(object):
    def __init__(self, max_angle=45, flip=True, scale_range=0.1, max_translation=np.array([1.0, 1.0, 0.5])):
        self.max_angle = max_angle
        self.flip = flip
        self.scale_range = scale_range
        self.max_translation = max_translation

    def merge(self, in_feat):
        # in_feat, Ts = self.augment(in_feat)
        
        min_coordinate = in_feat.C[:, 1:].min(dim=0)[0].int()
        in_feat_dense = in_feat.dense(
            min_coordinate=torch.IntTensor([*min_coordinate]))[0]

        in_feat_dense = torch.cat([in_feat_dense[t] for t in range(
            in_feat_dense.shape[0])], dim=0).unsqueeze(0)
        in_feat_merged = ME.to_sparse(in_feat_dense.float())
        coords = in_feat_merged.C.clone()
        # add min_coordinate back
        coords[:, 1:] += min_coordinate.reshape(1, -1)
        in_feat_merged = ME.SparseTensor(in_feat_merged.F, coords)
        return in_feat_merged
