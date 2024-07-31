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


import numpy as np


class WarmupCosine:
    def __init__(self, warmup_end, max_iter, factor_min):
        self.max_iter = max_iter
        self.warmup_end = warmup_end
        self.factor_min = factor_min

    def __call__(self, iter):
        if iter < self.warmup_end:
            factor = iter / self.warmup_end
        else:
            iter = iter - self.warmup_end
            max_iter = self.max_iter - self.warmup_end
            iter = (iter / max_iter) * np.pi
            factor = self.factor_min + 0.5 * (1 - self.factor_min) * (np.cos(iter) + 1)
        return factor
