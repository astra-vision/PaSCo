import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pasco.maskpls.interpolate import knn_up




class ASPP(nn.Module):
  '''
  ASPP
  '''
  def __init__(self, planes, dilations, norm_layer, act_layer):
    super().__init__()
    
    # ASPP Block
    self.conv_list = dilations
    self.conv1 = nn.ModuleList(
      [ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dil, dimension=3, bias=False) for dil in dilations])
    self.bn1 = nn.ModuleList([norm_layer(planes) for dil in dilations])
    self.conv2 = nn.ModuleList(
      [ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dil, dimension=3, bias=False) for dil in dilations])
    self.bn2 = nn.ModuleList([norm_layer(planes) for dil in dilations])
    self.relu = act_layer(inplace=True)
    self.pooling = ME.MinkowskiGlobalPooling()
    self.linear = ME.MinkowskiConvolution(planes * 2, planes, kernel_size=1, stride=1, dimension=3, bias=True)

  def forward(self, x_in):
    
    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
      
      
      
    x_out = self.relu(y + x_in)  # modified
    
    x_global = self.pooling(x_out)
    
    x_out_F = torch.cat([x_out.F, x_global.F.expand(x_out.F.shape[0], -1)], axis=1)
    x_out = ME.SparseTensor(
            x_out_F,
            coordinate_map_key=x_out.coordinate_map_key,
            coordinate_manager=x_out.coordinate_manager,
        )
    x_out = self.linear(x_out)
    return x_out



class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        # Global coords does not require coords_key
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid())
        self.pooling = ME.MinkowskiGlobalPooling()
 

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
 

        return ME.SparseTensor(
            y.F * x.F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        # return self.broadcast_mul(x, y)
    

class MinkEncoderDecoderModified(nn.Module):    
    """
    ResNet-like architecture using sparse convolutions
    """

    def __init__(self, input_dim, out_dim, channels, drop_path=0.0):
        super().__init__()

        cs = channels

        
        dpr = torch.linspace(0, drop_path, 16)  # stochastic depth decay rule
        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlockOriginal(cs[0], cs[1], ks=3, drop_path=dpr[0]),
            ResidualBlockOriginal(cs[1], cs[1], ks=3, drop_path=dpr[1]),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlockOriginal(cs[1], cs[2], ks=3, drop_path=dpr[2]),
            ResidualBlockOriginal(cs[2], cs[2], ks=3, drop_path=dpr[3]),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2),
            ResidualBlockOriginal(cs[2], cs[3], ks=3, drop_path=dpr[4]),
            ResidualBlockOriginal(cs[3], cs[3], ks=3, drop_path=dpr[5]),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2),
            ResidualBlockOriginal(cs[3], cs[4], ks=3, drop_path=dpr[6]),
            ResidualBlockOriginal(cs[4], cs[4], ks=3, drop_path=dpr[7]),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[5] + cs[3], cs[5], ks=3, drop_path=dpr[8]),
                    ResidualBlockOriginal(cs[5], cs[5], ks=3, drop_path=dpr[9]),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[6] + cs[2], cs[6], ks=3, drop_path=dpr[10]),
                    ResidualBlockOriginal(cs[6], cs[6], ks=3, drop_path=dpr[11]),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[7] + cs[1], cs[7], ks=3, drop_path=dpr[12]),
                    ResidualBlockOriginal(cs[7], cs[7], ks=3, drop_path=dpr[13]),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[8] + cs[0], cs[8], ks=3, drop_path=dpr[14]),
                    ResidualBlockOriginal(cs[8], cs[8], ks=3, drop_path=dpr[15]),
                ),
            ]
        )

        dims = {
            1: 64, 2:128, 4:128
        }
        in_dims = {
            1: cs[8], 2: cs[7], 4: cs[6]
        }
       

        self.sem_head = ME.MinkowskiConvolution(
                cs[-1], out_dim, kernel_size=1, dimension=3
            )
        
     

    def forward(self, x):
        x = ME.SparseTensor(x.F, x.C)
        # x0 = self.stem(x)
        x0 = x
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        out_feats = [y1, y2, y3, y4]
       
        return self.sem_head(out_feats[-1])


class MinkEncoderDecoderLite(nn.Module):
    """
    ResNet-like architecture using sparse convolutions
    """

    def __init__(self,  f):
        super().__init__()

        cs = [f, f, f, f, f, f, f, f, f]

        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlock(cs[0], cs[1], ks=3),
            ResidualBlock(cs[1], cs[1], ks=3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlock(cs[1], cs[2], ks=3),
            ResidualBlock(cs[2], cs[2], ks=3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2),
            ResidualBlock(cs[2], cs[3], ks=3),
            ResidualBlock(cs[3], cs[3], ks=3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2),
            ResidualBlock(cs[3], cs[4], ks=3),
            ResidualBlock(cs[4], cs[4], ks=3),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[5] + cs[3], cs[5], ks=3),
                    ResidualBlock(cs[5], cs[5], ks=3),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[6] + cs[2], cs[6], ks=3),
                    ResidualBlock(cs[6], cs[6], ks=3),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[7] + cs[1], cs[7], ks=3),
                    ResidualBlock(cs[7], cs[7], ks=3),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[8] + cs[0], cs[8], ks=3),
                    ResidualBlock(cs[8], cs[8], ks=3),
                ),
            ]
        )




    def forward(self, x):
        in_field = self.TensorField(x)

        x0 = self.stem(in_field.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        return y4



class MinkEncoderDecoder(nn.Module):
    """
    ResNet-like architecture using sparse convolutions
    """

    def __init__(self, cfg, data_cfg):
        super().__init__()

        n_classes = data_cfg.NUM_CLASSES

        input_dim = cfg.INPUT_DIM
        self.res = cfg.RESOLUTION
        self.knn_up = knn_up(cfg.KNN_UP)

        cs = cfg.CHANNELS
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(input_dim, cs[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlockOriginal(cs[0], cs[1], ks=3),
            ResidualBlockOriginal(cs[1], cs[1], ks=3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlockOriginal(cs[1], cs[2], ks=3),
            ResidualBlockOriginal(cs[2], cs[2], ks=3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2),
            ResidualBlockOriginal(cs[2], cs[3], ks=3),
            ResidualBlockOriginal(cs[3], cs[3], ks=3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2),
            ResidualBlockOriginal(cs[3], cs[4], ks=3),
            ResidualBlockOriginal(cs[4], cs[4], ks=3),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[5] + cs[3], cs[5], ks=3),
                    ResidualBlockOriginal(cs[5], cs[5], ks=3),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[6] + cs[2], cs[6], ks=3),
                    ResidualBlockOriginal(cs[6], cs[6], ks=3),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[7] + cs[1], cs[7], ks=3),
                    ResidualBlockOriginal(cs[7], cs[7], ks=3),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlockOriginal(cs[8] + cs[0], cs[8], ks=3),
                    ResidualBlockOriginal(cs[8], cs[8], ks=3),
                ),
            ]
        )

        self.sem_head = nn.Linear(cs[-1], 20)

        levels = [cs[-i] for i in range(4, 0, -1)]
        self.out_bnorm = nn.ModuleList([nn.BatchNorm1d(l) for l in levels])

    def forward(self, x):
        in_field = self.TensorField(x)

        x0 = self.stem(in_field.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        out_feats = [y1, y2, y3, y4]

        # vox2feat and apply batchnorm
        coors = [in_field.decomposed_coordinates for _ in range(len(out_feats))]
        coors = [[c * self.res for c in coors[i]] for i in range(len(coors))]
        bs = in_field.coordinate_manager.number_of_unique_batch_indices()
        vox_coors = [
            [l.coordinates_at(i) * self.res for i in range(bs)] for l in out_feats
        ]
        feats = [
            [
                bn(self.knn_up(vox_c, vox_f, pt_c))
                for vox_c, vox_f, pt_c in zip(vc, vf.decomposed_features, pc)
            ]
            for vc, vf, pc, bn in zip(vox_coors, out_feats, coors, self.out_bnorm)
        ]

        feats, coors, pad_masks = self.pad_batch(coors, feats)
        logits = self.sem_head(feats[-1])
        return feats, coors, pad_masks, logits

    def TensorField(self, x):
        """
        Build a tensor field from coordinates and features from the
        input batch
        The coordinates are quantized using the provided resolution
        """
        feat_tfield = ME.TensorField(
            features=torch.cat(x["feats"], 0).float(),
            coordinates=ME.utils.batched_coordinates(
                [i / self.res for i in x["pt_coord"]], dtype=torch.float
            ),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device="cuda",
        )
        return feat_tfield

    def pad_batch(self, coors, feats):
        """
        From a list of multi-level features create a list of batched tensors with
        features padded to the max number of points in the batch.

        returns:
            feats: List of batched feature Tensors per feature level
            coors: List of batched coordinate Tensors per feature level
            pad_masks: List of batched bool Tensors indicating padding
        """
        # get max number of points in the batch for each feature level
        maxs = [max([level.shape[0] for level in batch]) for batch in feats]
        # pad and batch each feature level in a single Tensor
        coors = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(coors)
        ]
        pad_masks = [
            torch.stack(
                [
                    F.pad(
                        torch.zeros_like(f[:, 0]), (0, maxs[i] - f.shape[0]), value=1
                    ).bool()
                    for f in batch
                ]
            )
            for i, batch in enumerate(feats)
        ]
        feats = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(feats)
        ]
        return feats, coors, pad_masks


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class BasicGenerativeDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=D,
                expand_coordinates=True
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )
     

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.F.shape[0],) + (1,) * (x.F.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output_F = x.F.div(keep_prob) * random_tensor
    output = ME.SparseTensor(output_F, 
                             coordinate_map_key=x.coordinate_map_key,
                             coordinate_manager=x.coordinate_manager,)
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    


class ResidualBlockOriginal(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3, drop_path=0., use_se=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            self.se = SELayer(outc, reduction=2)

    def forward(self, x):
        skip = self.downsample(x)
        y = self.net(x)
        if self.use_se:
            y = self.se(y)
        out = self.relu(y + skip)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3, drop_path=0., use_se=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.net = nn.Sequential(
            ME.MinkowskiBatchNorm(inc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1,
                                 dimension=D),
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                # ME.MinkowskiBatchNorm(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            self.se = SELayer(outc, reduction=2)

    def forward(self, x):
        skip = self.downsample(x)
        y = self.net(x)
        if self.use_se:
            y = self.se(y)
        out = self.relu(skip + y)
        return out

if __name__ == "__main__":
    n_channels = [32, 32, 64, 128, 256, 256, 128, 96, 96]
    model = MinkEncoderDecoderModified(input_dim=32, out_dim=20, channels=n_channels)
    state_dict = torch.load("/lustre/fsn1/projects/rech/kvd/uyl37fq/code/segcontrast_pretrain/lastepoch199_model_segment_contrast.pt", map_location='cpu')
    model.load_state_dict(state_dict["model"], strict=False)