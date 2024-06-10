# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
from time import time
from pasco.models.metrics import SSCMetrics

# Must be imported before large libs
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

import MinkowskiEngine as ME
from pasco.maskpls.mink import BasicConvolutionBlock, ResidualBlock


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer,
        act_layer,
        dropout_layer,
        drop_path_rates=[0.0, 0.0, 0.0],
        downsample=True,
        use_se_layer=False,
        dropout=0.0,
    ) -> None:
        super().__init__()
        print("enc use_se_layer", use_se_layer)
        self.downsample = downsample
        if self.downsample:
            # self.down = nn.Sequential(
            #    DepthwiseSeparableConvMultiheadsV2(
            #         in_channels, out_channels,
            #         n_heads=n_heads,
            #         kernel_size=2, stride=2
            #     ),
            #     norm_layer(out_channels),
            #     act_layer(),
            # )
            self.down = nn.Sequential(
                BasicConvolutionBlock(in_channels, out_channels, ks=2, stride=2),
                norm_layer(out_channels),
                act_layer(),
            )
            self.conv = nn.Sequential(
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[0]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[1]),
                ResidualBlock(out_channels, out_channels, drop_path=drop_path_rates[2]),
                dropout_layer(p=dropout),
            )
        else:
            self.conv = nn.Sequential(
                ResidualBlock(out_channels, out_channels),
                ResidualBlock(out_channels, out_channels),
                ResidualBlock(out_channels, out_channels),
                dropout_layer(p=dropout),
            )

    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        return self.conv(x)


class Encoder3DSepV2(nn.Module):

    def __init__(
        self,
        in_channels,
        f,
        norm_layer,
        act_layer,
        dropout_layer,
        dropouts,
        heavy_decoder=True,
        drop_path_rates=None,
        n_heads=1,
        use_se_layer=False,
    ):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        enc_ch = f

        self.enc_in_feats = ME.MinkowskiConvolution(
            in_channels, enc_ch[0], kernel_size=1, stride=1, dimension=3
        )

        if drop_path_rates is None:
            drop_path_rates = [0.0] * 12

        if not heavy_decoder:
            self.s1 = nn.Sequential(
                ResidualBlock(enc_ch[0], enc_ch[0]),
                ResidualBlock(enc_ch[0], enc_ch[0]),
                ResidualBlock(enc_ch[0], enc_ch[0]),
                nn.Identity(),
            )
            self.s1s2 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[0], enc_ch[1], ks=2, stride=2),
                norm_layer(enc_ch[1]),
                act_layer(),
                ResidualBlock(enc_ch[1], enc_ch[1]),
                ResidualBlock(enc_ch[1], enc_ch[1]),
                ResidualBlock(enc_ch[1], enc_ch[1]),
            )

            self.s2s4 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[1], enc_ch[2], ks=2, stride=2),
                norm_layer(enc_ch[2]),
                act_layer(),
                ResidualBlock(enc_ch[2], enc_ch[2]),
                ResidualBlock(enc_ch[2], enc_ch[2]),
                ResidualBlock(enc_ch[2], enc_ch[2]),
            )

            self.s4s8 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[2], enc_ch[3], ks=2, stride=2),
                norm_layer(enc_ch[3]),
                act_layer(),
                ResidualBlock(enc_ch[3], enc_ch[3]),
                ResidualBlock(enc_ch[3], enc_ch[3]),
                ResidualBlock(enc_ch[3], enc_ch[3]),
            )
        else:
            self.s1 = nn.Sequential(
                nn.Identity(),
            )
            self.s1s2 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[0], enc_ch[1], ks=2, stride=2),
                norm_layer(enc_ch[1]),
                act_layer(),
                dropout_layer(p=dropouts[-3]),
            )

            self.s2s4 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[1], enc_ch[2], ks=2, stride=2),
                norm_layer(enc_ch[2]),
                act_layer(),
                dropout_layer(p=dropouts[-2]),
            )

            self.s4s8 = nn.Sequential(
                BasicConvolutionBlock(enc_ch[2], enc_ch[3], ks=2, stride=2),
                norm_layer(enc_ch[3]),
                act_layer(),
                dropout_layer(p=dropouts[-1]),
            )

    def forward(self, in_feats):
        partial_in = self.enc_in_feats(in_feats)

        enc_s1 = self.s1(partial_in)
        enc_s2 = self.s1s2(enc_s1)
        enc_s4 = self.s2s4(enc_s2)
        enc_s8 = self.s4s8(enc_s4)
        features = [enc_s1, enc_s2, enc_s4, enc_s8]

        return features
