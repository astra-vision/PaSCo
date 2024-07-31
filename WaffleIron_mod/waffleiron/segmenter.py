# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
from .backbone import WaffleIron
from .embedding import Embedding


class Segmenter(nn.Module):
    def __init__(self, input_channels, feat_channels, nb_class, depth, grid_shape):
        super().__init__()
        # Embedding layer
        self.embed = Embedding(input_channels, feat_channels)
        # WaffleIron backbone
        self.waffleiron = WaffleIron(feat_channels, depth, grid_shape)
        # Classification layer
        self.classif = nn.Conv1d(feat_channels, nb_class, 1)

    def forward(self, feats, cell_ind, occupied_cell, neighbors):
        embedding = self.embed(feats, neighbors) # TODO: use this one also
        tokens = self.waffleiron(embedding, cell_ind, occupied_cell)
        return embedding, tokens, self.classif(tokens)
        # return self.classif(tokens)
