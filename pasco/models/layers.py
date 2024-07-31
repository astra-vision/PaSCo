import MinkowskiEngine as ME
import torch
import torch.nn as nn
from torch.nn import functional as F


# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import math
from typing import Union

import torch
from torch.nn import Parameter


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, norm_layer=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        layers = []
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            layers.append(nn.Linear(n, k))
            if i < num_layers - 1:
                if norm_layer:
                    layers.append(nn.BatchNorm1d(k))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class CAM(nn.Module):
    def __init__(self, planes, reduction=2):
        super(CAM, self).__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiConvolution(
                planes, planes // reduction, kernel_size=1, stride=1, dimension=3
            ),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                planes // reduction, planes, kernel_size=1, stride=1, dimension=3
            ),
            ME.MinkowskiSigmoid(),
        )
        self.pooling = ME.MinkowskiMaxPooling(kernel_size=7, stride=1, dimension=3)

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return x * y


class PointwiseConvMultiheadsDense(nn.Module):
    def __init__(self, inplanes, planes, n_heads=1):
        super(PointwiseConvMultiheads, self).__init__()
        self.n_heads = n_heads
        self.planes = planes
        self.pointwise_head = nn.Conv3d(
            inplanes, planes, kernel_size=1, dimension=3, bias=False
        )

        in_dim_per_head = inplanes // n_heads
        out_dim_per_head = planes // n_heads
        kernel_mask = torch.zeros_like(self.pointwise_head.kernel)
        import pdb

        pdb.set_trace()
        for i in range(n_heads):
            in_idx = torch.arange(i * in_dim_per_head, (i + 1) * in_dim_per_head)
            out_idx = torch.arange(i * out_dim_per_head, (i + 1) * out_dim_per_head)

            xs, ys = torch.meshgrid(in_idx, out_idx)
            kernel_mask[xs, ys] = 1
        self.pointwise_head.kernel = nn.Parameter(
            self.pointwise_head.kernel * kernel_mask
        )

    def forward(self, x):
        x = self.pointwise_head(x)
        return x


class PointwiseConvMultiheads(nn.Module):
    def __init__(self, inplanes, planes, n_heads=1):
        super(PointwiseConvMultiheads, self).__init__()
        self.n_heads = n_heads
        self.planes = planes
        self.pointwise_head = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=3, bias=False
        )

        in_dim_per_head = inplanes // n_heads
        out_dim_per_head = planes // n_heads
        kernel_mask = torch.zeros_like(self.pointwise_head.kernel)
        for i in range(n_heads):
            in_idx = torch.arange(i * in_dim_per_head, (i + 1) * in_dim_per_head)
            out_idx = torch.arange(i * out_dim_per_head, (i + 1) * out_dim_per_head)

            xs, ys = torch.meshgrid(in_idx, out_idx)
            kernel_mask[xs, ys] = 1
        self.pointwise_head.kernel = nn.Parameter(
            self.pointwise_head.kernel * kernel_mask
        )

    def forward(self, x):
        x = self.pointwise_head(x)
        return x


class DepthwiseSeparableConvMultiheadsV2(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, n_heads=1, stride=1, bias=False
    ):
        super(DepthwiseSeparableConvMultiheadsV2, self).__init__()
        self.convs = nn.ModuleDict()
        self.n_heads = n_heads
        for i_head in range(n_heads):
            self.convs[str(i_head)] = ME.MinkowskiConvolution(
                inplanes // n_heads,
                planes // n_heads,
                kernel_size=kernel_size,
                stride=stride,
                dimension=3,
                bias=bias,
            )

    def forward(self, x):

        x_heads = []
        n_feats = x.F.shape[1] // self.n_heads

        for i_head in range(self.n_heads):
            # with torch.cuda.stream(streams[i_head]):
            x_head_F = x.F[:, i_head * n_feats : (i_head + 1) * n_feats]
            x_head = ME.SparseTensor(
                x_head_F,
                tensor_stride=x.tensor_stride,
                coordinate_manager=x.coordinate_manager,
                coordinate_map_key=x.coordinate_map_key,
            )
            x_head = self.convs[str(i_head)](x_head)
            x_heads.append(x_head)
        if self.n_heads > 1:
            x = ME.cat(x_heads)
        else:
            x = x_heads[0]
        return x


class DepthwiseSeparableConvMultiheads(nn.Module):
    def __init__(self, planes, kernel_size=3, n_heads=1):
        super(DepthwiseSeparableConvMultiheads, self).__init__()
        self.n_heads = n_heads
        self.planes = planes
        self.depthwise = ME.MinkowskiChannelwiseConvolution(
            planes, kernel_size=kernel_size, dimension=3, bias=False
        )
        self.pointwise_head = PointwiseConvMultiheads(planes, planes, n_heads=n_heads)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise_head(x)
        return x


class DepthwiseSeparableResBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, norm_layer, n_heads, skip_conv=True, use_se_layer=False
    ):
        super(DepthwiseSeparableResBlock, self).__init__()
        self.skip_conv = skip_conv
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=1, dimension=3, bias=False
        )

        self.norm1 = norm_layer(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dimension=3, bias=False
        )
        self.norm2 = norm_layer(planes)
        self.act = ME.MinkowskiReLU()

        self.use_se_layer = use_se_layer
        if use_se_layer:
            self.se = SELayer(planes, reduction=2)

        if inplanes != planes and self.skip_conv:
            self.skip = ME.MinkowskiConvolution(
                inplanes, planes, kernel_size=1, stride=1, dimension=3, bias=False
            )
        else:
            self.skip = None

    def forward(self, x):
        residual = x
        if self.skip is not None:
            residual = self.skip(residual)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_se_layer:
            out = self.se(out)

        if self.skip_conv:
            out += residual

        out = self.act(out)

        return out


class DepthwiseSeparableConvMultiheadsBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, n_heads=1):
        super(DepthwiseSeparableConvMultiheadsBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConvMultiheads(
            planes, kernel_size=kernel_size, n_heads=n_heads
        )
        self.conv2 = DepthwiseSeparableConvMultiheads(
            planes, kernel_size=kernel_size, n_heads=n_heads
        )
        self.relu = ME.MinkowskiReLU(inplace=False)

    def forward(self, x_in):
        x = self.conv1(self.relu(x_in))
        x = self.conv2(self.relu(x))
        return x


class DepthwiseUpsample(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=2, n_heads=1):
        super(DepthwiseUpsample, self).__init__()
        self.unpooling = ME.MinkowskiPoolingTranspose(
            kernel_size=kernel_size, stride=stride, dimension=3, expand_coordinates=True
        )
        self.resize = PointwiseConvMultiheads(inplanes, planes, n_heads=n_heads)

    def forward(self, x_in):
        x = self.unpooling(x_in)
        x = self.resize(x)

        return x


class ARBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        act_layer,
        stride=1,
        dilation=1,
        dimension=-1,
    ):
        super(ARBlock, self).__init__()

        self.conv_l1 = ME.MinkowskiConvolution(
            inplanes,
            planes,
            kernel_size=(3, 1, 3),
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.conv_l2 = ME.MinkowskiConvolution(
            inplanes,
            planes,
            kernel_size=(1, 3, 3),
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )

        self.conv_r1 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=(1, 3, 3),
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.conv_r2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=(3, 1, 3),
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )

        self.norm_l1 = norm_layer(planes)
        self.norm_l2 = norm_layer(planes)
        self.norm_r1 = norm_layer(planes)
        self.norm_r2 = norm_layer(planes)
        self.act_l1 = act_layer()
        self.act_l2 = act_layer()
        self.act_r1 = act_layer()
        self.act_r2 = act_layer()

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.norm_l1(x_l)
        x_l = self.act_l1(x_l)

        x_l = self.conv_l2(x_l)
        x_l = self.norm_l2(x_l)
        x_l = self.act_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.norm_r1(x_r)
        x_r = self.act_r1(x_r)

        x_r = self.conv_r2(x_r)
        x_r = self.norm_r2(x_r)
        x_r = self.act_r2(x_r)

        return x_l + x_r


class DenseResBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        padding,
        norm_layer,
        act_layer,
        bias=False,
        n_infers=1,
    ):
        super(DenseResBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            groups=n_infers,
            bias=bias,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = norm_layer(planes)
        self.conv2 = nn.Conv3d(
            inplanes,
            planes,
            groups=n_infers,
            bias=bias,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm2 = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.act(out)

        return out


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        act_layer,
        use_se_layer=False,
        stride=1,
        dilation=1,
        dimension=-1,
    ):
        super(BasicResBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )

        self.norm1 = norm_layer(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm2 = norm_layer(planes)
        self.act = act_layer()

        self.use_se_layer = use_se_layer
        if use_se_layer:
            self.se = SELayer(planes, reduction=2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_se_layer:
            out = self.se(out)

        out += residual
        out = self.act(out)

        return out


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=1
    ):
        super().__init__()
        self.depth = nn.Conv3d(
            inplanes,
            inplanes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=inplanes,
        )
        self.point = nn.Conv3d(
            inplanes,
            planes,
            stride=1,
            kernel_size=1,
            padding=0,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return x


class SPCDense3D(nn.Module):
    def __init__(self, init_size=16, n_infers=1):
        super(SPCDense3D, self).__init__()

        conv_layer = nn.Conv3d
        ### Completion sub-network
        bias = False  # False
        chs = [init_size, init_size * 1, init_size * 1, init_size * 1]
        self.a_conv1 = nn.Sequential(
            conv_layer(chs[1], chs[1], 3, 1, padding=1, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv2 = nn.Sequential(
            conv_layer(chs[1], chs[1], 3, 1, padding=1, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv3 = nn.Sequential(
            conv_layer(chs[1], chs[1], 5, 1, padding=2, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv4 = nn.Sequential(
            conv_layer(chs[1], chs[1], 7, 1, padding=3, bias=bias, groups=n_infers),
            nn.ReLU(),
        )

        self.a_conv5 = nn.Sequential(
            conv_layer(chs[1], chs[1], 3, 1, padding=1, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv6 = nn.Sequential(
            conv_layer(chs[1], chs[1], 5, 1, padding=2, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv7 = nn.Sequential(
            conv_layer(chs[1], chs[1], 7, 1, padding=3, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.ch_conv1 = nn.Sequential(
            nn.Conv3d(
                chs[1], chs[0], kernel_size=1, stride=1, bias=bias, groups=n_infers
            ),
            nn.ReLU(),
        )

        self.res_1 = nn.Sequential(
            conv_layer(chs[0], chs[0], 3, 1, padding=1, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.res_2 = nn.Sequential(
            conv_layer(chs[0], chs[0], 5, 1, padding=2, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.res_3 = nn.Sequential(
            conv_layer(chs[0], chs[0], 7, 1, padding=3, bias=bias, groups=n_infers),
            nn.ReLU(),
        )

    def forward(self, x_dense):
        ### Completion sub-network by dense convolution

        x1 = self.a_conv1(x_dense)
        x2 = self.a_conv2(x1)
        x3 = self.a_conv3(x1)
        x4 = self.a_conv4(x1)
        t1 = x2 + x3 + x4
        x5 = self.a_conv5(t1)
        x6 = self.a_conv6(t1)
        x7 = self.a_conv7(t1)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7
        y0 = self.ch_conv1(x)
        y1 = self.res_1(x_dense)
        y2 = self.res_2(x_dense)
        y3 = self.res_3(x_dense)
        x = x_dense + y0 + y1 + y2 + y3
        return x


class SPCDense3DConcat(nn.Module):
    def __init__(self, init_size=16, n_infers=1):
        super(SPCDense3DConcat, self).__init__()

        conv_layer = nn.Conv3d

        ### Completion sub-network
        bias = False  # False
        chs = [init_size, init_size * 1, init_size * 1, init_size * 1]
        self.a_conv1 = nn.Sequential(
            conv_layer(chs[1], chs[1], 3, 1, padding=1, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv2 = nn.Sequential(
            conv_layer(chs[1], chs[1], 3, 1, padding=1, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv3 = nn.Sequential(
            conv_layer(chs[1], chs[1], 5, 1, padding=2, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.a_conv4 = nn.Sequential(
            conv_layer(chs[1], chs[1], 7, 1, padding=3, bias=bias, groups=n_infers),
            nn.ReLU(),
        )

        self.a_conv5 = nn.Sequential(
            nn.Conv3d(chs[1] * 3, chs[1], 3, 1, padding=1, bias=bias), nn.ReLU()
        )
        self.a_conv6 = nn.Sequential(
            nn.Conv3d(chs[1] * 3, chs[1], 5, 1, padding=2, bias=bias), nn.ReLU()
        )
        self.a_conv7 = nn.Sequential(
            nn.Conv3d(chs[1] * 3, chs[1], 7, 1, padding=3, bias=bias), nn.ReLU()
        )
        self.ch_conv1 = nn.Sequential(
            nn.Conv3d(chs[1] * 7, chs[0], kernel_size=1, stride=1, bias=bias), nn.ReLU()
        )

        self.res_1 = nn.Sequential(
            conv_layer(chs[0], chs[0], 3, 1, padding=1, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.res_2 = nn.Sequential(
            conv_layer(chs[0], chs[0], 5, 1, padding=2, bias=bias, groups=n_infers),
            nn.ReLU(),
        )
        self.res_3 = nn.Sequential(
            conv_layer(chs[0], chs[0], 7, 1, padding=3, bias=bias, groups=n_infers),
            nn.ReLU(),
        )

    def forward(self, x_dense):
        ### Completion sub-network by dense convolution
        x1 = self.a_conv1(x_dense)
        x2 = self.a_conv2(x1)
        x3 = self.a_conv3(x1)
        x4 = self.a_conv4(x1)
        t1 = torch.cat((x2, x3, x4), 1)
        x5 = self.a_conv5(t1)
        x6 = self.a_conv6(t1)
        x7 = self.a_conv7(t1)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), 1)
        y0 = self.ch_conv1(x)
        y1 = self.res_1(x_dense)
        y2 = self.res_2(x_dense)
        y3 = self.res_3(x_dense)
        x = x_dense + y0 + y1 + y2 + y3
        return x


class SPCDense3Dv2(nn.Module):
    def __init__(self, init_size=16):
        super(SPCDense3Dv2, self).__init__()

        conv_layer = nn.Conv3d

        ### Completion sub-network
        bias = False
        act = nn.Identity()
        chs = [init_size, init_size * 1, init_size * 1, init_size * 1]
        self.a_conv1 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_1 = nn.BatchNorm3d(chs[1])

        self.a_conv2 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_2 = nn.BatchNorm3d(chs[1])
        self.a_conv3 = nn.Sequential(
            conv_layer(chs[1], chs[1], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias), act
        )
        self.bn_3 = nn.BatchNorm3d(chs[1])
        self.a_conv4 = nn.Sequential(
            conv_layer(chs[1], chs[1], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias), act
        )
        self.bn_4 = nn.BatchNorm3d(chs[1])

        self.a_conv5 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_5 = nn.BatchNorm3d(chs[1])
        self.a_conv6 = nn.Sequential(
            conv_layer(chs[1], chs[1], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias), act
        )
        self.bn_6 = nn.BatchNorm3d(chs[1])
        self.a_conv7 = nn.Sequential(
            conv_layer(chs[1], chs[1], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias), act
        )
        self.bn_7 = nn.BatchNorm3d(chs[1])
        self.ch_conv1 = nn.Sequential(
            nn.Conv3d(chs[1], chs[0], kernel_size=1, stride=1, bias=bias), act
        )
        self.bn_ch_conv1 = nn.BatchNorm3d(chs[0])

        self.res_1 = nn.Sequential(
            conv_layer(chs[0], chs[0], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_res_1 = nn.BatchNorm3d(chs[0])
        self.res_2 = nn.Sequential(
            conv_layer(chs[0], chs[0], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias), act
        )
        self.bn_res_2 = nn.BatchNorm3d(chs[0])
        self.res_3 = nn.Sequential(
            conv_layer(chs[0], chs[0], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias), act
        )
        self.bn_res_3 = nn.BatchNorm3d(chs[0])

    def forward(self, x_dense):
        ### Completion sub-network by dense convolution

        x1 = F.relu(self.bn_1(self.a_conv1(x_dense)))

        x2 = F.relu(self.bn_2(self.a_conv2(x1)))
        x3 = F.relu(self.bn_3(self.a_conv3(x1)))
        x4 = F.relu(self.bn_4(self.a_conv4(x1)))

        t1 = x2 + x3 + x4

        x5 = F.relu(self.bn_5(self.a_conv5(t1)))
        x6 = F.relu(self.bn_6(self.a_conv6(t1)))
        x7 = F.relu(self.bn_7(self.a_conv7(t1)))

        x = x1 + x2 + x3 + x4 + x5 + x6 + x7
        y0 = F.relu(self.bn_ch_conv1(self.ch_conv1(x)))
        y1 = F.relu(self.bn_res_1(self.res_1(x_dense)))
        y2 = F.relu(self.bn_res_2(self.res_2(x_dense)))
        y3 = F.relu(self.bn_res_3(self.res_3(x_dense)))
        x = x1 + y0 + y1 + y2 + y3
        # return F.relu(x)
        return x


class ExpandCoords(nn.Module):
    def __init__(self, init_size=16, n_infers=1):
        super(ExpandCoords, self).__init__()

        conv_layer = ME.MinkowskiConvolution
        ### Completion sub-network
        bias = False
        chs = [init_size, init_size * 1, init_size * 1, init_size * 1]
        self.a_conv1 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), bias=bias, expand_coordinates=True)
        )
        self.bn_1 = ME.MinkowskiBatchNorm(chs[1])

        self.a_conv2 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), bias=bias, expand_coordinates=True)
        )
        self.bn_2 = ME.MinkowskiBatchNorm(chs[1])
        self.a_conv3 = nn.Sequential(
            conv_layer(chs[1], chs[1], (5, 5, 3), bias=bias, expand_coordinates=True)
        )
        self.bn_3 = ME.MinkowskiBatchNorm(chs[1])
        self.a_conv4 = nn.Sequential(
            conv_layer(chs[1], chs[1], (7, 7, 5), bias=bias, expand_coordinates=True)
        )
        self.bn_4 = ME.MinkowskiBatchNorm(chs[1])

        self.a_conv5 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), bias=bias, expand_coordinates=True)
        )
        self.bn_5 = ME.MinkowskiBatchNorm(chs[1])
        self.a_conv6 = nn.Sequential(
            conv_layer(chs[1], chs[1], (5, 5, 3), bias=bias, expand_coordinates=True)
        )
        self.bn_6 = ME.MinkowskiBatchNorm(chs[1])
        self.a_conv7 = nn.Sequential(
            conv_layer(chs[1], chs[1], (7, 7, 5), bias=bias, expand_coordinates=True)
        )
        self.bn_7 = ME.MinkowskiBatchNorm(chs[1])
        self.ch_conv1 = nn.Sequential(
            conv_layer(chs[1], chs[0], kernel_size=1, stride=1, bias=bias)
        )

        self.res_1 = nn.Sequential(
            conv_layer(chs[0], chs[0], (3, 3, 1), bias=bias, expand_coordinates=True)
        )
        self.bn_res_1 = ME.MinkowskiBatchNorm(chs[0])
        self.res_2 = nn.Sequential(
            conv_layer(chs[0], chs[0], (5, 5, 3), bias=bias, expand_coordinates=True)
        )
        self.bn_res_2 = ME.MinkowskiBatchNorm(chs[0])
        self.res_3 = nn.Sequential(
            conv_layer(chs[0], chs[0], (7, 7, 5), bias=bias, expand_coordinates=True)
        )
        self.bn_res_3 = ME.MinkowskiBatchNorm(chs[0])

    def forward(self, x_dense):
        ### Completion sub-network by dense convolution

        x1 = self.bn_1(self.a_conv1(x_dense))

        x2 = self.bn_2(self.a_conv2(x1))
        x3 = self.bn_3(self.a_conv3(x1))
        x4 = self.bn_4(self.a_conv4(x1))

        t1 = x2 + x3 + x4
        t1 = F.relu(t1)

        x5 = self.bn_5(self.a_conv5(t1))
        x6 = self.bn_6(self.a_conv6(t1))
        x7 = self.bn_7(self.a_conv7(t1))

        x = F.relu(x1 + x2 + x3 + x4 + x5 + x6 + x7)
        y0 = self.ch_conv1(x)
        y1 = self.bn_res_1(self.res_1(x_dense))
        y2 = self.bn_res_2(self.res_2(x_dense))
        y3 = self.bn_res_3(self.res_3(x_dense))
        x = x1 + y0 + y1 + y2 + y3
        return F.relu(x)


if __name__ == "__main__":

    pointwise_dense = PointwiseConvMultiheadsDense(8, 8, n_heads=2)
    x = torch.randn(2, 8, 4, 4, 4)
    x = pointwise_dense(x)
